"""_v14_upstream.py — L12/L13/L14/L15 单-cell donor replacement（因果上游定位）。只读。

只读。0 训练 / 0 反传 / 0 优化器 / 0 改 checkpoint-模型源码-loss-assigner-augmentation-config /
0 test inference / 0 submission / 0 P2-OASA-crop-1536 / 0 架构实验。不保存任何被改过的模型。

Frozen（与 V13 逐字相同）：checkpoint / preprocessing / eval mode / images / GT groups
（G1=38, G4=194）/ donor mapping（同类别∧small∧成功∧G4 → min|Δsqrt(area)| → tie-break (stem,gt)）。
Donor 的 L12–L15 向量直接取自 V11 的 _v11_layers.npz（**未重新选择 donor**）。

每层用**自己的** stride 独立计算 GT-center cell（禁止复用 L17/L23 坐标），并加边界断言。
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
SMALL = 32.0
PATCH_LAYERS = (12, 13, 14, 15)
DISP = 4                                   # 位移控制：该层自身坐标系中的 +4 cells


def main():
    cfg = yaml_model_load(str(ROOT / "configs/yolo11m_sepstem.yaml"))
    model = DetectionModel(cfg, nc=12, verbose=False)
    w, _ = attempt_load_one_weight(str(CKPT)); model.load(w)
    model.model[-1].stride = model.stride
    model.eval(); model.model[-1].train()
    d = model.model[-1]
    Wcls = d.cv3[0][-1].weight.detach().clone().view(12, -1)
    bcls = d.cv3[0][-1].bias.detach().clone()

    cap, patch = {}, {"spec": None, "h": None}
    d.cv3[0][-1].register_forward_pre_hook(lambda m, i: patch.__setitem__("h", i[0].detach()))

    def mk(i):
        def h(m, inp, out):
            o = out[0] if isinstance(out, (list, tuple)) else out
            if not torch.is_tensor(o):
                return out
            sp = patch["spec"]
            if sp and sp["layer"] == i:
                o2 = o.clone()
                yy, xx = sp["y"], sp["x"]
                assert 0 <= yy < o2.shape[2] and 0 <= xx < o2.shape[3], \
                    f"L{i} cell out of bounds: ({yy},{xx}) vs {tuple(o2.shape[2:])}"
                o2[0, :, yy, xx] = torch.as_tensor(sp["vec"], dtype=o2.dtype, device=o2.device)
                cap[i] = o2.detach()
                return o2
            cap[i] = o.detach()
            return out
        return h
    for li in PATCH_LAYERS + (23,):
        model.model[li].register_forward_hook(mk(li))
    print(f"hooks: {PATCH_LAYERS} patchable + L23 recorded")

    # ---- 各层 shape / stride（§4）----
    with torch.no_grad():
        model(torch.zeros(1, 5, 1280, 1280))
    print("\n§4 各层坐标系：")
    for li in PATCH_LAYERS:
        t = cap[li]
        print(f"  L{li}: shape={tuple(t.shape)}  stride={1280//t.shape[-1]}")

    Z = np.load(OUT / "_v11_layers.npz")

    # ---- 组定义（与 V10–V13 相同）----
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

    def grp(k):
        return ("G1" if k in E1 else "G4" if (k in v2rec and k not in E1 and k not in CTRL
                and v2rec[k]["native_area"] < 1024 and succ.get(k)) else None)

    ids = json.loads((OUT / "_v11_ids.json").read_text(encoding="utf-8"))
    idx_of = {g: {k: i for i, k in enumerate(ids[g])} for g in ids}
    pool = [dict(k=k, i=i, cls=int(v2rec[k]["cls"]) if "cls" in v2rec[k] else -1,
                 sq=float(np.sqrt(v2rec[k]["native_area"]))) for k, i in idx_of["G4"].items()]

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

    rows = []
    order = []
    for i in range(len(ds)):
        lb = ds.labels[i]; bbi = lb.get("bboxes")
        if bbi is None or len(bbi) == 0:
            continue
        st = Path(ds.im_files[i]).stem
        has1 = any(f"{st}#{j}" in E1 for j in range(len(bbi)))
        has4 = any(grp(f"{st}#{j}") == "G4" for j in range(len(bbi)))
        if has1 or has4:
            order.append((0 if has1 else 1, i))
    order.sort()
    print(f"  待处理 {len(order)} 图（含 G1 的 {sum(1 for a,_ in order if a==0)} 张优先）", flush=True)
    for _pri, i in order:
        lab = ds[i]; bb = lab.get("bboxes"); cl = lab.get("cls")
        if bb is None or len(bb) == 0:
            continue
        stem = Path(ds.im_files[i]).stem
        tg = []
        for j in range(len(bb)):
            sq = float(np.sqrt(max(float(bb[j][2]) * 1280 * float(bb[j][3]) * 1280, 0)))
            g = grp(f"{stem}#{j}")
            if g:
                tg.append((j, g, sq, int(cl[j])))
        if not tg:
            continue
        img = lab["img"].float().div(255.0)
        img = img.unsqueeze(0) if img.ndim == 3 else img
        with torch.no_grad():
            model(img)
        patch["spec"] = None
        base_h = patch["h"].clone(); base_L23 = cap[23].clone()
        hh3, ww3 = base_L23.shape[2:]
        y23 = x23 = 0
        for j, g, sq, c in tg:
            cx, cy = float(bb[j][0]) * 1280, float(bb[j][1]) * 1280
            y23 = int(min(max(cy / (1280 / hh3), 0), hh3 - 1))
            x23 = int(min(max(cx / (1280 / ww3), 0), ww3 - 1))
            b_lg = float(Wcls[c] @ base_h[0, :, y23, x23] + bcls[c])
            r = dict(key=f"{stem}#{j}", grp=g, cls=c, sq=sq, base_logit=b_lg,
                     base_prob=float(1 / (1 + np.exp(-b_lg))))
            cand = [dd for dd in pool if dd["cls"] == c]
            r["n_donor"] = len(cand)
            if cand:
                dsel = min(cand, key=lambda dd: (abs(dd["sq"] - sq), dd["k"]))
                r["donor"] = dsel["k"]
                LAY = PATCH_LAYERS if g == "G1" else (12, 15)      # G4 至少 L12/L15
                for li in LAY:
                    t = cap[li]; hh, ww = t.shape[2:]
                    yy = int(min(max(cy / (1280 / hh), 0), hh - 1))
                    xx = int(min(max(cx / (1280 / ww), 0), ww - 1))
                    assert 0 <= yy < hh and 0 <= xx < ww, f"L{li} bounds"
                    dv = torch.from_numpy(Z[f"L{li}_G4"][dsel["i"]]).clone()
                    ov_ = t[0, :, yy, xx]

                    def do(tag, vec):
                        patch["spec"] = dict(layer=li, vec=vec, y=yy, x=xx)
                        with torch.no_grad():
                            model(img)
                        patch["spec"] = None
                        lg = float(Wcls[c] @ patch["h"][0, :, y23, x23] + bcls[c])
                        r[f"{tag}_logit"] = lg
                        r[f"{tag}_dlogit"] = lg - b_lg
                        r[f"{tag}_dp"] = float(1 / (1 + np.exp(-lg))) - r["base_prob"]
                        r[f"{tag}_L23norm"] = float(cap[23][0, :, y23, x23].norm())

                    do(f"L{li}_raw", dv)
                    do(f"L{li}_norm", dv / (dv.norm() + 1e-12) * ov_.norm())
                    do(f"L{li}_disp", dv)          # 占位，见下（用位移坐标重做）
                    # 真正的位移控制：在**该层自身坐标系**中 +4 cells
                    patch["spec"] = dict(layer=li, vec=dv, y=yy, x=min(xx + DISP, ww - 1))
                    with torch.no_grad():
                        model(img)
                    patch["spec"] = None
                    lg = float(Wcls[c] @ patch["h"][0, :, y23, x23] + bcls[c])
                    r[f"L{li}_disp_dlogit"] = lg - b_lg
                    r[f"L{li}_disp_x"] = min(xx + DISP, ww - 1)
            rows.append(r)
        print(f"  {stem}: {len(tg)} target", flush=True)
        if _pri == 0:
            (OUT / "_v14_upstream.json").write_text(json.dumps(rows, ensure_ascii=False),
                                                    encoding="utf-8")
    (OUT / "_v14_upstream.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {OUT/'_v14_upstream.json'} ({len(rows)} target)")


if __name__ == "__main__":
    main()
