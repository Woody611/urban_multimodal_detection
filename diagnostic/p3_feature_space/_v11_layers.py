"""_v11_layers.py — layer 9→23 逐层 trajectory（定位 FIRST STABLE DIVERGENCE）。只读。

只读。只碰 train/val。0 训练 / 0 改模型-loss-assigner-增强 / 0 核心文件改动 / 0 碰 test-submission-online。
样本与 V10 **完全一致**（同 seed、同 image 采样、同 group 定义）。
采样与 V10 **完全一致**（GT-center 单 cell；每层用自己的 map 宽度算 sc=1280/ww）。
"""
import json
import random
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
SEED, N_TRAIN_IMG, SMALL = 20260929, 300, 32.0     # ← 与 V10 逐字相同
LAYERS = list(range(9, 24))


def main():
    cfg = yaml_model_load(str(ROOT / "configs/yolo11m_sepstem.yaml"))
    model = DetectionModel(cfg, nc=12, verbose=False)
    w, _ = attempt_load_one_weight(str(CKPT)); model.load(w)
    model.model[-1].stride = model.stride
    model.eval(); model.model[-1].train()
    cur = {}
    for li in LAYERS:
        def mk(i):
            def h(mm, inp, out):
                o = out[0] if isinstance(out, (list, tuple)) else out
                if torch.is_tensor(o):
                    cur[i] = o.detach()
            return h
        model.model[li].register_forward_hook(mk(li))
    print(f"hooks: layers {LAYERS}")

    # ---- 组定义（与 V10 相同口径）----
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

    def grp_of(dom, key, sq):
        if dom == "T1":
            return "TS" if sq < SMALL else None
        k = key
        return ("G1" if k in E1 else "G2" if k in CTRL else
                "G4" if (k in v2rec and k not in E1 and k not in CTRL
                          and v2rec[k]["native_area"] < 1024 and succ.get(k)) else None)

    data = yaml_load(str(ROOT / "data/processed/rgbid_split_train/dataset.yaml"))
    store = {}                      # layer -> {grp -> list of vecs}
    ids = {"G1": [], "G4": [], "TS": []}
    for li in LAYERS:
        store[li] = {g: [] for g in ("G1", "G4", "TS")}
    for dom, tr in (("T1", True), ("V1", False)):
        ov = dict(imgsz=1280, task="detect", rect=False, cache=False, single_cls=False,
                  classes=None, fraction=1.0, channels=5, use_simotm="RGBID",
                  object_scale_aug=False, mosaic=0.0, mixup=0.0, copy_paste=0.0,
                  degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
                  flipud=0.0, fliplr=0.5, augment=False)
        ds = build_yolo_dataset(get_cfg(overrides=ov),
                                str((Path(data["path"]) / (data["train"] if tr else data["val"])).resolve()),
                                8, data, mode="val", use_simotm="RGBID",
                                pairs_rgb_ir=["visible", "infrared"],
                                pairs_rgb_depth=["visible", "depth"])
        idxs = (list(range(0, len(ds), max(1, len(ds) // N_TRAIN_IMG)))[:N_TRAIN_IMG]
                if dom == "T1" else range(len(ds)))
        for i in idxs:
            random.seed(SEED + i); np.random.seed(SEED + i)   # ← 与 V10 相同
            lab = ds[i]; bb = lab.get("bboxes"); cl = lab.get("cls")
            if bb is None or len(bb) == 0:
                continue
            stem = Path(ds.im_files[i]).stem
            sq_all = [float(np.sqrt(max(float(bb[j][2]) * 1280 * float(bb[j][3]) * 1280, 0)))
                      for j in range(len(bb))]
            tg = [(j, grp_of(dom, f"{stem}#{j}", sq_all[j])) for j in range(len(bb))]
            tg = [(j, g) for j, g in tg if g in ("G1", "G4", "TS")]  # G2 不入库
            if not tg:
                continue
            img = lab["img"].float().div(255.0)
            img = img.unsqueeze(0) if img.ndim == 3 else img
            cur.clear()
            with torch.no_grad():
                model(img)
            if len(cur) < len(LAYERS):
                continue
            for j, g in tg:
                cx, cy = float(bb[j][0]) * 1280, float(bb[j][1]) * 1280
                for li in LAYERS:
                    t = cur[li]
                    hh, ww = t.shape[2:]
                    sc = 1280.0 / ww
                    x = int(min(max(cx / sc, 0), ww - 1)); y = int(min(max(cy / sc, 0), hh - 1))
                    store[li][g].append(t[0, :, y, x].numpy().astype(np.float32))
                ids[g].append(f"{stem}#{j}")
        print(f"  [{dom}] done")
    np.savez_compressed(OUT / "_v11_layers.npz",
                        **{f"L{li}_{g}": np.stack(store[li][g]) for li in LAYERS
                           for g in ("G1", "G4", "TS")})
    (OUT / "_v11_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    print("组 n:", {g: len(ids[g]) for g in ids})
    print(f"[saved] {OUT/'_v11_layers.npz'}")


if __name__ == "__main__":
    main()
