"""_v13_dose.py — L17 spatial dose-response（R0–R3）。只读 causal diagnostic。

只读。0 训练 / 0 反传 / 0 改 checkpoint-模型源码-loss-assigner-augmentation-config /
0 test inference / 0 submission / 0 新架构实验。不保存任何被改过的模型。

Frozen（与 V12 逐字相同）：checkpoint / preprocessing / eval mode / images / G1(38)/G4(194) /
GT-center mapping / donor selection（同类别∧small∧成功∧G4 → min|Δsqrt(area)| → tie-break (stem,gt)）。

几何：L17 stride 32 ⇒ 40×40 @1280。R0=1cell, R1=3×3=9, R2=5×5=25, R3=7×7=49，严格 clamp 在边界内。
Mode A = raw donor；Mode B = 逐 cell norm-matched（每个被替换 cell 各自匹配其原 norm）。
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
RADII = {0: 0, 1: 1, 2: 2, 3: 3}
SESSION_RNG = 12345          # random-donor 固定 seed（与 V12 同值）


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
                for (yy, xx), vec in sp["cells"]:
                    o2[0, :, yy, xx] = torch.as_tensor(vec, dtype=o2.dtype, device=o2.device)
                cap[i] = o2.detach()
                return o2
            cap[i] = o.detach()
            return out
        return h
    for li in (17, 23):
        model.model[li].register_forward_hook(mk(li))
    print("hooks: L17(17) patchable + L23(23) recorded")

    # ---- TS 的 L23 质心方向（用于 §11 cosine）----
    Z = np.load(OUT / "_v11_layers.npz")
    muTS23 = Z["L23_TS"].astype(np.float64).mean(0)
    muTS23 /= (np.linalg.norm(muTS23) + 1e-12)
    print(f"  TS L23 质心已建（dim={muTS23.shape[0]}）")

    # ---- 组定义（与 V10/V11/V12 相同）----
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

    # ---- donor 池（规则与 V12 完全一致）----
    ids = json.loads((OUT / "_v11_ids.json").read_text(encoding="utf-8"))
    idx_of = {g: {k: i for i, k in enumerate(ids[g])} for g in ids}
    donor_pool = [dict(k=k, i=i, cls=int(v2rec[k]["cls"]) if "cls" in v2rec[k] else -1,
                       sq=float(np.sqrt(v2rec[k]["native_area"])))
                  for k, i in idx_of["G4"].items()]

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
    for i in range(len(ds)):
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
        for j, g, sq, c in tg:
            cx, cy = float(bb[j][0]) * 1280, float(bb[j][1]) * 1280
            # ⚠️ L17 是 40×40、L23/head 输入是 160×160 —— 两个坐标系必须分开算
            hh, ww = cap[17].shape[2:]                    # L17: 40x40
            y0, x0 = int(min(max(cy / (1280 / hh), 0), hh - 1)), int(min(max(cx / (1280 / ww), 0), ww - 1))
            hh3, ww3 = base_L23.shape[2:]                 # L23 / cv3 输入: 160x160
            y23 = int(min(max(cy / (1280 / hh3), 0), hh3 - 1))
            x23 = int(min(max(cx / (1280 / ww3), 0), ww3 - 1))
            b_lg = float(Wcls[c] @ base_h[0, :, y23, x23] + bcls[c])
            r = dict(key=f"{stem}#{j}", grp=g, cls=c, sq=sq, base_logit=b_lg,
                     base_prob=float(1 / (1 + np.exp(-b_lg))),
                     L23_cos_base=float(cap[23][0, :, y23, x23].numpy().astype(np.float64) @ muTS23
                                        / (np.linalg.norm(cap[23][0, :, y23, x23].numpy()) + 1e-12)),
                     L23_norm_base=float(cap[23][0, :, y23, x23].norm()))
            # donor（规则固定）
            cand = [dd for dd in donor_pool if dd["cls"] == c]
            r["n_donor"] = len(cand)
            if cand:
                dsel = min(cand, key=lambda dd: (abs(dd["sq"] - sq), dd["k"]))
                dv = torch.from_numpy(Z["L17_G4"][dsel["i"]]).clone()
                rnd = cand[int(np.random.default_rng(SESSION_RNG).integers(len(cand)))]
                rv = torch.from_numpy(Z["L17_G4"][rnd["i"]]).clone()

                def cells_for(radius, vec, mode, base_t):
                    cs = []
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            yy, xx = y0 + dy, x0 + dx
                            if 0 <= yy < hh and 0 <= xx < ww:
                                v = vec
                                if mode == "norm":
                                    ov_ = base_t[0, :, yy, xx]
                                    v = vec / (vec.norm() + 1e-12) * ov_.norm()
                                cs.append(((yy, xx), v))
                    return cs

                def do(tag, layer, vec, radius, mode, base_t):
                    sp = dict(layer=layer, cells=cells_for(radius, vec, mode, base_t))
                    patch["spec"] = sp
                    with torch.no_grad():
                        model(img)
                    patch["spec"] = None
                    lg = float(Wcls[c] @ patch["h"][0, :, y23, x23] + bcls[c])
                    v23 = cap[23][0, :, y23, x23].numpy()
                    r[f"{tag}_logit"] = lg
                    r[f"{tag}_dlogit"] = lg - b_lg
                    r[f"{tag}_dprob"] = float(1 / (1 + np.exp(-lg))) - r["base_prob"]
                    r[f"{tag}_L23_cos"] = float(v23.astype(np.float64) @ muTS23
                                               / (np.linalg.norm(v23) + 1e-12))
                    r[f"{tag}_L23_norm"] = float(np.linalg.norm(v23))
                    r[f"{tag}_ncells"] = len(sp["cells"])

                if g == "G1":
                    for R in RADII:
                        do(f"R{R}_raw", 17, dv, RADII[R], "raw", base_L23)
                        do(f"R{R}_norm", 17, dv, RADII[R], "norm", base_L23)
                        do(f"R{R}_rnd", 17, rv, RADII[R], "raw", base_L23)
                    for R in (0, 2):                       # displaced：固定方向 +4 / +5
                        for dsh in (4, 5):
                            yy, xx = y0, x0 + dsh
                            if xx >= ww:
                                yy, xx = y0, x0 - dsh
                            if 0 <= yy < hh and 0 <= xx < ww:
                                sp = dict(layer=17, cells=[((yy, xx), dv)])
                                patch["spec"] = sp
                                with torch.no_grad():
                                    model(img)
                                patch["spec"] = None
                                lg = float(Wcls[c] @ patch["h"][0, :, y23, x23] + bcls[c])
                                r[f"disp{dsh}_logit"] = lg
                                r[f"disp{dsh}_dlogit"] = lg - b_lg
                else:                                       # G4：R0 与 R2
                    for R in (0, 2):
                        do(f"R{R}_raw", 17, dv, RADII[R], "raw", base_L23)
            rows.append(r)
        print(f"  {stem}: {len(tg)} target", flush=True)
    (OUT / "_v13_dose.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {OUT/'_v13_dose.json'} ({len(rows)} target)")


if __name__ == "__main__":
    main()
