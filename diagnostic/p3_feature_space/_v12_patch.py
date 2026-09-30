"""_v12_patch.py — READ-ONLY LOCAL ACTIVATION INTERVENTION（L16 / L17）。

只读。0 训练 / 0 反传 / 0 优化器 / 0 改 checkpoint-模型源码-loss-assigner-augmentation-config /
0 test inference / 0 submission / 0 P2-OASA-crop-1536。不保存任何被改过的模型。

Donor 选择规则（**在跑 intervention 之前固定，不使用 logit / feature similarity / outcome**）：
    同类别 ∧ small(sq<32) ∧ 成功(best_same_class_iou≥0.5) ∧ 来自 G4 或 TS
    → 取 |sqrt(area_donor) − sqrt(area_G1)| 最小者；并列时按 (stem, gt) 字典序
    （规格允许 "preferably similar native sqrt(area)"；除此之外不看任何 outcome 量）
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "diagnostic/small_object_cause_v2"))

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, yaml_model_load  # noqa: E402
from ultralytics.utils import yaml_load  # noqa: E402

OUT = Path(__file__).resolve().parent
CKPT = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
V2 = ROOT / "diagnostic/small_object_cause_v2"
SEED, N_TRAIN_IMG, SMALL = 20260929, 300, 32.0
PATCH_LAYERS = {"L16": 16, "L17": 17}


def main():
    cfg = yaml_model_load(str(ROOT / "configs/yolo11m_sepstem.yaml"))
    model = DetectionModel(cfg, nc=12, verbose=False)
    w, _ = attempt_load_one_weight(str(CKPT)); model.load(w)
    model.model[-1].stride = model.stride
    model.eval(); model.model[-1].train()
    d = model.model[-1]
    Wcls = d.cv3[0][-1].weight.detach().clone().view(12, -1)   # (12,256)
    bcls = d.cv3[0][-1].bias.detach().clone()                  # (12,)

    cap, patch = {}, {"spec": None, "h": None}
    # H = cv3[0][-1] 的输入（精确分解：logit_c = W_c·h + b_c）
    d.cv3[0][-1].register_forward_pre_hook(lambda m, i: patch.__setitem__("h", i[0].detach()))
    for li in (16, 17, 23):
        def mk(i):
            def h(m, inp, out):
                o = out[0] if isinstance(out, (list, tuple)) else out
                if not torch.is_tensor(o):
                    return out
                if patch["spec"] and patch["spec"]["layer"] == i:
                    p = patch["spec"]
                    o2 = o.clone()
                    v = torch.as_tensor(p["vec"], dtype=o2.dtype, device=o2.device)
                    o2[0, :, p["y"], p["x"]] = v
                    cap[i] = o2.detach()
                    return o2
                cap[i] = o.detach()
                return out
            return h
        model.model[li].register_forward_hook(mk(li))
    print("hooks: L16(16)/L17(17)/L23(23) 输出 + cv3[0][-1] 输入")

    # ---------- 组定义（与 V10/V11 相同）----------
    v3 = json.loads((V2 / "_v3_analysis.json").read_text(encoding="utf-8"))
    E1 = set(v3["E1"]); CTRL = set(v[0] for v in v3["controls"].values())
    v2rec = {f"{r['image_id']}#{r['gt_id']}": r for r in
             json.loads((V2 / "_v2_records.json").read_text(encoding="utf-8"))["records"]}
    from official_eval import (load_split, read_pred_txt, apply_max_boxes,
                               norm_xywh_to_xyxy, box_iou_np, MAX_BOXES_PER_IMAGE)
    succ = {}
    for _i, s_, w_, h_, gb, gc in load_split(
            ROOT / "data/processed/rgbid_split_train/images/val/visible",
            ROOT / "data/processed/rgbid_split_train/labels/val/visible")[0]:
        if gb is None or len(gb) == 0:
            continue
        _, pr, _ = read_pred_txt(ROOT / f"diagnostic/sepstem_clahe/best_full/results/{s_}.txt")
        pr = apply_max_boxes(pr, MAX_BOXES_PER_IMAGE)
        pb, pc = (norm_xywh_to_xyxy(pr, w_, h_) if len(pr) else
                  (np.zeros((0, 4), np.float32), np.zeros(0, int)))
        for j in range(len(gb)):
            sm = np.where(pc == gc[j])[0] if len(pc) else np.zeros(0, int)
            succ[f"{s_}#{j}"] = (float(box_iou_np(gb[j:j+1], pb[sm])[0].max()) if len(sm) else 0.0) >= 0.5

    def grp(dom, k, sq):
        if dom == "T1":
            return "TS" if sq < SMALL else None
        return ("G1" if k in E1 else "G4" if (k in v2rec and k not in E1 and k not in CTRL
                and v2rec[k]["native_area"] < 1024 and succ.get(k)) else None)

    # ---------- 一次性取 donor 池的 L16/L17 向量（来自 V11，无需新 forward）----------
    Z = np.load(OUT / "_v11_layers.npz")
    ids = json.loads((OUT / "_v11_ids.json").read_text(encoding="utf-8"))
    # G1/G4/TS 的 (key -> idx)
    idx_of = {g: {k: i for i, k in enumerate(ids[g])} for g in ids}
    # donor 池 = G4 ∪ TS（同类别、small、成功）—— 需要 cls 与 sq：从 v11 的 npz 维度无法取，改用 v2rec/数据集
    # 这里用 donor 的 key 反查 cls/sq：G4 来自 v2rec；TS 来自 train（cls 需另取）
    donor_pool = []
    for k, i in idx_of["G4"].items():
        donor_pool.append(dict(k=k, src="G4", i=i, cls=int(v2rec[k]["cls"] if "cls" in v2rec[k] else -1),
                               sq=float(np.sqrt(v2rec[k]["native_area"]))))
    # TS 的 cls/sq 从 v11 抽取时的数据集重算成本高；本轮 donor 只取 G4（同为 val success，口径更干净）
    print(f"donor 池 = G4 {len(donor_pool)}（同类别/成功/small；TS 因需重算 cls 未纳入，规则已在写码前固定）")

    # ---------- 逐图 forward + 干预 ----------
    data = yaml_load(str(ROOT / "data/processed/rgbid_split_train/dataset.yaml"))
    ov = dict(imgsz=1280, task="detect", rect=False, cache=False, single_cls=False, classes=None,
              fraction=1.0, channels=5, use_simotm="RGBID", object_scale_aug=False, mosaic=0.0,
              mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
              perspective=0.0, flipud=0.0, fliplr=0.5, augment=False)
    ds = build_yolo_dataset(get_cfg(overrides=ov),
                            str((Path(data["path"]) / data["val"]).resolve()), 8, data,
                            mode="val", use_simotm="RGBID",
                            pairs_rgb_ir=["visible", "infrared"],
                            pairs_rgb_depth=["visible", "depth"])

    def cellof(li, cx, cy):
        t = cap[li]; hh, ww = t.shape[2:]; sc = 1280.0 / ww
        return int(min(max(cy / sc, 0), hh - 1)), int(min(max(cx / sc, 0), ww - 1))

    def run(img, spec):
        patch["spec"] = spec
        with torch.no_grad():
            model(img)
        patch["spec"] = None
        return cap[23], patch["h"]

    def logit_of(h, c, y, x):
        """精确分解：logit_c = W_c · h + b_c；h 取 **GT 中心 cell**（不是角落 [0,0]）。"""
        return float(Wcls[c] @ h[0, :, y, x] + bcls[c])

    rows = []
    for i in range(len(ds)):
        lab = ds[i]; bb = lab.get("bboxes"); cl = lab.get("cls")
        if bb is None or len(bb) == 0:
            continue
        stem = Path(ds.im_files[i]).stem
        tg = []
        for j in range(len(bb)):
            sq = float(np.sqrt(max(float(bb[j][2]) * 1280 * float(bb[j][3]) * 1280, 0)))
            g = grp("V1", f"{stem}#{j}", sq)
            if g:
                tg.append((j, g, sq, int(cl[j])))
        if not tg:
            continue
        img = lab["img"].float().div(255.0)
        img = img.unsqueeze(0) if img.ndim == 3 else img
        # baseline
        run(img, None)
        base_L23 = cap[23].clone(); base_h = patch["h"].clone()
        for j, g, sq, c in tg:
            cx, cy = float(bb[j][0]) * 1280, float(bb[j][1]) * 1280
            y17, x17 = cellof(17, cx, cy); y16, x16 = cellof(16, cx, cy)
            y23, x23 = cellof(23, cx, cy)
            b_lg = logit_of(base_h, c, y23, x23)
            r = dict(key=f"{stem}#{j}", grp=g, cls=c, sq=sq,
                     base_logit=b_lg, base_prob=float(1 / (1 + np.exp(-b_lg))),
                     y17=y17, x17=x17, y16=y16, x16=x16,
                     L17_norm=float(cap[17][0, :, y17, x17].norm()),
                     L23_norm=float(cap[23][0, :, y23, x23].norm()))
            v17 = cap[17][0, :, y17, x17].clone()
            v16 = cap[16][0, :, y16, x16].clone()
            # donor（规则固定：|Δsq| 最小；并列按 key 字典序）
            cand = [dd for dd in donor_pool if dd["cls"] == c]
            r["n_donor_cand"] = len(cand)
            if cand:
                dsel = min(cand, key=lambda dd: (abs(dd["sq"] - sq), dd["k"]))
                dv17 = torch.from_numpy(Z[f"L17_{dsel['src']}"][dsel["i"]]).clone()
                dv16 = torch.from_numpy(Z[f"L16_{dsel['src']}"][dsel["i"]]).clone()
                r["donor"] = dsel["k"]
                # 随机 donor（固定 seed，合法池内随机）
                rnd = cand[int(np.random.default_rng(12345).integers(len(cand)))]
                rv17 = torch.from_numpy(Z[f"L17_{rnd['src']}"][rnd["i"]]).clone()
                INT = {
                    "L16_zero": (16, torch.zeros_like(v16), y16, x16),
                    "L17_zero": (17, torch.zeros_like(v17), y17, x17),
                    "L16_proto": (16, dv16, y16, x16),
                    "L17_proto": (17, dv17, y17, x17),
                    "L17_proto_normmatch": (17, dv17 / (dv17.norm() + 1e-12) * v17.norm(), y17, x17),
                    "L17_displaced": (17, dv17, min(y17 + 3, cap[17].shape[2] - 1), x17),
                    "L17_random_donor": (17, rv17, y17, x17),
                }
                for nm, (li, vec, yy, xx) in INT.items():
                    run(img, dict(layer=li, vec=vec, y=yy, x=xx))
                    lg = logit_of(patch["h"], c, y23, x23)
                    r[f"{nm}_logit"] = lg
                    r[f"{nm}_dlogit"] = lg - b_lg
                    r[f"{nm}_dprob"] = float(1 / (1 + np.exp(-lg))) - r["base_prob"]
                # 恢复 baseline 供下一个目标使用
                run(img, None)
            rows.append(r)
        print(f"  {stem}: {len(tg)} target", flush=True)

    (OUT / "_v12_patch.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {OUT/'_v12_patch.json'}  ({len(rows)} target)")


if __name__ == "__main__":
    main()
