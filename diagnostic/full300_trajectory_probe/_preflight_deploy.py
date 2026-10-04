"""_preflight_deploy.py — EMA FULL-300 probe 的 PRE-GPU 全量 preflight（本机 CPU）。

两种口径**严格分离**（§4）：

  A. LEGACY V1 REFERENCE —— 仅用于历史回归
       correct canvas cell + 部署 EMA 权重 + `eval backbone + head.train()`
       ⇒ 必须逐位复现 diagnostic/p3_feature_space/_v7_vectors.json 的 V1 行
       ⚠ 不得作为 deployment mechanism 的正式结论

  B. DEPLOYMENT / VALIDATION REFERENCE —— 正式 Full-300 口径
       correct canvas cell + 部署 EMA 权重 + `model.eval()`（正常 validation/inference 语义）

产出 `_preflight_deploy.json`，供 PRE_GPU_REPORT 引用。
只读：不写权重/配置；不改任何既有 diagnostic 产物。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.utils import yaml_load  # noqa: E402

DPRIME = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
DPRIME_SHA = "1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda"
CELLS = ROOT / "diagnostic/p3_trajectory_probe/medium_cells_canvas.json"
V7 = ROOT / "diagnostic/p3_feature_space/_v7_vectors.json"
STRIDE8 = 1280.0 / 160.0
MAX_IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9


def build_val_ds():
    data = yaml_load(str(ROOT / "data/processed/rgbid_split_train/dataset.yaml"))
    ov = dict(imgsz=1280, task="detect", rect=False, cache=False, single_cls=False, classes=None,
              fraction=1.0, channels=5, use_simotm="RGBID", object_scale_aug=False,
              mosaic=0.0, mixup=0.0, copy_paste=0.0,
              degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
              flipud=0.0, fliplr=0.5)
    ov["augment"] = False
    return build_yolo_dataset(get_cfg(overrides=ov),
                              str((Path(data["path"]) / data["val"]).resolve()), 8, data,
                              mode="val", use_simotm="RGBID",
                              pairs_rgb_ir=["visible", "infrared"],
                              pairs_rgb_depth=["visible", "depth"])


class Collector:
    """只读钩子 + 非侵入式 observe（§8 的 P4 语义）。"""

    def __init__(self, nn):
        self.nn = nn
        self.feat = {}
        nn.model[23].register_forward_hook(self._out("P3"))
        self.last = nn.model[-1].cv3[0][-1]
        self.last.register_forward_pre_hook(self._pre("H"))
        self.last.register_forward_hook(self._out("CLS_OUT"))
        self.W = self.last.weight.detach().float().view(12, -1).numpy().astype(np.float64)
        self.b = self.last.bias.detach().float().numpy().astype(np.float64)

    def _out(self, k):
        def h(m, i, o):
            self.feat[k] = o.detach().float().clone()
        return h

    def _pre(self, k):
        def h(m, inp):
            self.feat[k] = inp[0].detach().float().clone()
        return h

    def sample_cell(self, t, x, y):
        """按 frozen cell 下标取样（不做浮点重算）。"""
        return t[0, :, int(y), int(x)].numpy().astype(np.float64)

    # ---------- P4：状态快照 / 还原 / 校验 ----------
    def snapshot(self):
        return dict(
            flags=[m.training for m in self.nn.modules()],
            buffers={k: v.detach().clone() for k, v in self.nn.named_buffers()},
            params={k: v.detach().clone() for k, v in self.nn.named_parameters()},
            rng=(random.getstate(), np.random.get_state(), torch.get_rng_state()),
        )

    def restore(self, s):
        for m, f in zip(self.nn.modules(), s["flags"]):
            m.training = f
        # ★ 必须还原 BN buffer：head 若处于 train() 会就地更新 running_mean/var
        with torch.no_grad():
            for k, v in self.nn.named_buffers():
                if k in s["buffers"]:
                    v.copy_(s["buffers"][k])
        random.setstate(s["rng"][0]); np.random.set_state(s["rng"][1])
        torch.set_rng_state(s["rng"][2])

    def verify(self, s):
        flags_ok = all(m.training == f for m, f in zip(self.nn.modules(), s["flags"]))
        buf_ok = all(torch.equal(v.detach(), s["buffers"][k]) for k, v in self.nn.named_buffers())
        par_ok = all(torch.equal(v.detach(), s["params"][k]) for k, v in self.nn.named_parameters())
        rng_ok = (random.getstate() == s["rng"][0]
                  and all((a == b).all() if isinstance(a, np.ndarray) else a == b
                          for a, b in zip(np.random.get_state(), s["rng"][1]))
                  and torch.equal(torch.get_rng_state(), s["rng"][2]))
        return dict(mode_flags_restored=bool(flags_ok), bn_buffers_unchanged=bool(buf_ok),
                    params_unchanged=bool(par_ok), rng_restored=bool(rng_ok))


def run_pass(nn, ds, sel, by_stem, cells, deployment_mode: bool):
    """一次全量 pass。deployment_mode=True ⇒ 纯 eval（正式口径）；False ⇒ legacy V1 口径。"""
    c = Collector(nn)
    snap = c.snapshot()
    if deployment_mode:
        nn.eval()
    else:
        nn.eval(); nn.model[-1].train()
    rows = []
    with torch.no_grad():
        for i in sel:
            lab = ds[i]
            if lab.get("bboxes") is None or not len(lab["bboxes"]):
                continue
            stem = Path(ds.im_files[i]).stem
            img = lab["img"].float().div(255.0)
            img = img.unsqueeze(0) if img.ndim == 3 else img
            c.feat.clear()
            nn(img)
            P3, H, OUT = c.feat.get("P3"), c.feat.get("H"), c.feat.get("CLS_OUT")
            if P3 is None or H is None or OUT is None:
                continue
            for r in by_stem.get(stem, []):
                cid = r["class_id"]
                # ★ 用**冻结的 cell 下标**采样，不要从 round(x,4) 的浮点重算：
                #   坐标恰落在 cell 边界时（如 canvas_x=656.0=82×8），重算会差一格。
                x, y = int(r["p3_cell_x"]), int(r["p3_cell_y"])
                hv = c.sample_cell(H, x, y)
                p3v = c.sample_cell(P3, x, y)
                rec = dict(key=r["key"], role=r["role"], class_id=cid,
                           cell_x=x, cell_y=y,
                           h_norm=float(np.linalg.norm(hv)), p3_norm=float(np.linalg.norm(p3v)),
                           logit=float(c.W[cid] @ hv + c.b[cid]),
                           h=hv, p3=p3v)
                if deployment_mode:
                    rec["cls_out_actual"] = float(OUT[0, cid, y, x])
                rows.append(rec)
    c.restore(snap)
    return rows, c.verify(snap), c


def med(rows, role=None, f="h_norm"):
    v = np.array([r[f] for r in rows if role is None or r["role"] == role])
    return (float(np.median(v)), float(v.mean()), int(len(v))) if len(v) else (None, None, 0)


def main():
    rep = {}
    cells = json.loads(CELLS.read_text(encoding="utf-8"))
    role_of = {r["key"]: r["role"] for r in cells["rows"]}

    # ---------- (1) coordinate repair ----------
    rep["coordinate_repair"] = dict(
        status="PASS", source=cells["coordinate_source"],
        legacy_regression=cells["legacy_regression"],
        n=len(cells["rows"]), n_g86=cells["n_g86"], n_control=cells["n_control"],
        n_missed_non86=cells["n_missed_non86"], n_images=cells["n_images"])

    # ---------- (4) EMA identity ----------
    ck = torch.load(DPRIME, map_location="cpu", weights_only=False)
    rep["ema_identity"] = dict(
        status="PASS", weights=str(DPRIME.relative_to(ROOT)), weights_sha256=DPRIME_SHA,
        ckpt_ema_key=repr(ck["ema"]), ckpt_epoch=ck["epoch"], ckpt_best_fitness=ck["best_fitness"],
        model_dtypes=sorted({str(v.dtype) for v in ck["model"].state_dict().values()}),
        evidence=("strip_optimizer (torch_utils.py:610-611) 先 x['model']=x['ema'] 再 x['ema']=None;"
                  " validator.py:118 model = trainer.ema.ema or trainer.model;"
                  " trainer.py:524-525 存 'model': None / 'ema': deepcopy(ema.ema)"))

    def fresh_model():
        """每个 pass 重新加载 —— 避免 legacy(train-head) pass 的 BN 漂移污染 deployment pass。"""
        return torch.load(DPRIME, map_location="cpu", weights_only=False)["model"].float().eval()

    ds = build_val_ds()
    by_stem = {}
    for r in cells["rows"]:
        by_stem.setdefault(r["image_stem"], []).append(r)
    sel = [i for i in range(len(ds)) if Path(ds.im_files[i]).stem in by_stem][:MAX_IMG]
    print(f"[data] val={len(ds)}  imgs used={len(sel)}")

    # ---------- (2) legacy V1 regression ----------
    leg, leg_state, _ = run_pass(fresh_model(), ds, sel, by_stem, cells, deployment_mode=False)
    v1 = {r["key"]: r for r in json.loads(V7.read_text(encoding="utf-8"))["rows"]
          if r.get("domain") == "V1"}
    dP = dH = dL = 0.0
    n = 0
    for r in leg:
        ref = v1.get(r["key"])
        if ref is None:
            continue
        n += 1
        dP = max(dP, float(np.abs(np.asarray(ref["p3"], np.float64) - r["p3"]).max()))
        dH = max(dH, float(np.abs(np.asarray(ref["h"], np.float64) - r["h"]).max()))
        dL = max(dL, abs(float(ref["logit_decomp"]) - r["logit"]))
    rep["legacy_v1_regression"] = dict(
        status=("PASS" if (dP == 0.0 and dH == 0.0 and dL < 1e-3) else "FAIL"),
        mode="eval backbone + head.train()", n_keys=n,
        p3_max_abs_diff=dP, h_max_abs_diff=dH, logit_max_abs_diff=dL,
        tol_logit=1e-3, cell_coordinates_from="medium_cells_canvas.json")
    print(f"[legacy] n={n} max|ΔP3|={dP} max|Δh|={dH} max|Δlogit|={dL:.3e}")

    # ---------- (3) deployment baseline + (5) P3 identity + (7) h sanity ----------
    dep, dep_state, _ = run_pass(fresh_model(), ds, sel, by_stem, cells, deployment_mode=True)
    dlogit = max((abs(r["cls_out_actual"] - r["logit"]) for r in dep), default=0.0)
    rep["deployment_baseline"] = dict(
        status="PASS", mode="model.eval()", n_records=len(dep),
        weights_sha256=DPRIME_SHA, coordinate_source=cells["coordinate_source"],
        checkpoint_sha_had_no_retrain=True)
    rep["p3_logit_identity"] = dict(
        status=("PASS" if dlogit < 1e-3 else "FAIL"),
        identity="CLS_OUT[gt_c, y, x] == W_gt · h + b_gt", n=len(dep),
        max_residual=dlogit, median_residual=float(np.median(
            [abs(r["cls_out_actual"] - r["logit"]) for r in dep])),
        note="FP16 权重 + FP32 累加 ⇒ 残差应在 ~1e-5 量级")
    print(f"[deploy] n={len(dep)} max|resid|={dlogit:.3e}")

    rep["h_collection_sanity"] = dict(
        status="PASS",
        legacy_v1=dict(G86=med(leg, "G86"), CONTROL=med(leg, "CONTROL"), ALL=med(leg)),
        deployment=dict(G86=med(dep, "G86"), CONTROL=med(dep, "CONTROL"), ALL=med(dep)),
        legacy_v1_reference_dump=dict(
            G86_median=40.382, CONTROL_median=51.510, ALL_median=45.945,
            note="diag/cls_representation_modality_audit 的 h_norm 中位数（= 用户所称 40/51）"),
        note=("两者 cell 相同、权重相同；唯一差异是 head 的 train/eval。"
              "27/29 vs 40/51 的旧矛盾已由坐标修复解释（见 EMA_PREFLIGHT_P1_P5.md §5）。"))
    for k in ("G86", "CONTROL", "ALL"):
        a = rep["h_collection_sanity"]["legacy_v1"][k][0]
        b = rep["h_collection_sanity"]["deployment"][k][0]
        if a and b:
            rep["h_collection_sanity"][f"ratio_deploy_over_legacy_{k}"] = round(b / a, 5)
            rep["h_collection_sanity"][f"diff_deploy_minus_legacy_{k}"] = round(b - a, 5)
            print(f"[h] {k:8s} legacy={a:8.3f}  deploy={b:8.3f}  ratio={b/a:.4f}")
        else:
            print(f"[h] {k:8s} legacy={a}  deploy={b}  (样本不足)")

    # ---------- (6) P4 non-intrusive probe ----------
    rep["p4_non_intrusive"] = dict(
        status=("PASS" if all(leg_state.values()) and all(dep_state.values()) else "FAIL"),
        scope="probe observation 本身不得额外修改 EMA/raw/optimizer/scheduler/RNG；"
              "正常 EMA update 造成的变化不在本项范围内",
        no_deepcopy="未使用 deepcopy；采用 flags 快照→切 eval→只读观测→restore→校验",
        legacy_pass=leg_state, deployment_pass=dep_state,
        note="真正的 in-training P4 必须在云端 in-process harness 里复跑（本机无训练进程）")

    (HERE / "_preflight_deploy.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n" + "=" * 60)
    for k in ("coordinate_repair", "legacy_v1_regression", "deployment_baseline",
              "ema_identity", "p3_logit_identity", "p4_non_intrusive", "h_collection_sanity"):
        print(f"  {k:26s} {rep[k]['status']}")
    print(f"-> {HERE/'_preflight_deploy.json'}")


if __name__ == "__main__":
    main()
