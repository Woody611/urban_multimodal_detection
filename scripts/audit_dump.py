"""scripts/audit_dump.py — E7 推理一致性审计：固定 10 张图的逐图逐框 dump。

用途：本地与云端跑**同一个脚本、同一批固定图片**，各自产出 JSON，直接 diff 定位
第一处差异。脚本完全复刻 scripts/predict.py 的推理路径（LoadImagesAndVideos +
同一 LetterBox + 同一 non_max_suppression + 同一 scale_boxes），不做任何额外处理。

固定 10 张图的选取规则（确定性，两边必然一致）：
    取 val split 全部图片按 stem 升序排序，取下标 0,40,80,...,360 共 10 张。

输出：JSON，含环境指纹 + 每张图的几何/计数/前 20 个框。

用法（本地与云端相同）:
    python scripts/audit_dump.py --out audit_local.json
    python scripts/audit_dump.py --out audit_cloud.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.data.loaders import LoadImagesAndVideos  # noqa: E402
from ultralytics.utils.ops import non_max_suppression, scale_boxes  # noqa: E402

WEIGHTS = "runs/urban_multimodal_det_e7_yolo11m_rgbd_1024/weights/best.pt"
SOURCE = "data/processed/depth_split_train/images/val/visible"
IMGSZ, CONF, IOU, MAX_DET, MAX_BOXES, BATCH = 1024, 0.001, 0.7, 300, 100, 16
USE_SIMOTM, CHANNELS = "RGBD", 4
PAIRS = ["visible", "depth"]
N_PICK, STRIDE_PICK = 10, 40


def sha256(path: Path) -> str:
    if not path.exists():
        return "<missing>"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_stems(vis_dir: Path):
    stems = sorted(p.stem for p in vis_dir.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    return [stems[i] for i in range(0, len(stems), STRIDE_PICK)][:N_PICK], len(stems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--iou", type=float, default=IOU)
    ap.add_argument("--conf", type=float, default=CONF)
    args = ap.parse_args()

    src = (PROJECT_ROOT / args.source).resolve()
    wp = (PROJECT_ROOT / args.weights).resolve()
    picks, n_total = pick_stems(src)

    env = {
        "cwd": str(Path.cwd()),
        "python": sys.version.split()[0],
        "impl": platform.python_implementation(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "ultralytics_file": __import__("ultralytics").__file__,
        "weights_path": str(wp),
        "weights_size": wp.stat().st_size if wp.exists() else None,
        "weights_sha256": sha256(wp),
        "code_sha256": {f: sha256(PROJECT_ROOT / f) for f in (
            "ultralytics/data/augment.py", "ultralytics/data/base.py",
            "ultralytics/engine/model.py", "ultralytics/utils/metrics.py",
            "scripts/predict.py")},
        "params": dict(weights=str(wp), source=str(src), imgsz=IMGSZ, conf=args.conf,
                       iou=args.iou, max_det=MAX_DET, max_boxes=MAX_BOXES, batch=BATCH,
                       use_simotm=USE_SIMOTM, channels=CHANNELS, pairs=PAIRS,
                       tta=False, half=False, agnostic_nms=False),
        "n_val_images": n_total,
        "picked_stems": picks,
    }

    model = YOLO(str(wp))
    nn_model = model.model.to("cpu").eval()
    nc = int(getattr(nn_model, "nc", 12))
    stride = int(nn_model.stride.max()) if getattr(nn_model, "stride", None) is not None else 32
    letterbox = LetterBox(new_shape=(IMGSZ, IMGSZ), auto=False, scaleFill=False,
                          scaleup=True, center=True, stride=stride)

    # ---- 只取固定 10 张（LoadImagesAndVideos 顺序即文件顺序，用 stems 过滤）----
    loader = LoadImagesAndVideos(str(src), batch=BATCH, use_simotm=USE_SIMOTM,
                                 imgsz=IMGSZ, pairs_rgb_ir=PAIRS, pairs_rgb_depth=PAIRS)
    env["loader_ni"] = int(loader.ni)

    want = set(picks)
    records = {}
    for paths, imgs, _info in loader:
        sel = [(i, p) for i, p in enumerate(paths) if Path(p).stem in want]
        if not sel:
            continue
        keep = [i for i, _ in sel]
        sub_imgs = [imgs[i] for i in keep]
        orig_shapes = [im.shape[:2] for im in sub_imgs]

        # 复刻 predict.py::_preprocess_batch
        arrs = []
        for im in sub_imgs:
            im_c = np.ascontiguousarray(letterbox(image=im).transpose(2, 0, 1))
            arrs.append(np.concatenate([im_c[:3][::-1], im_c[3:]], axis=0))  # [R,G,B,D]
        x = torch.from_numpy(np.stack(arrs)).float() / 255.0

        with torch.no_grad():
            preds = nn_model(x)
        # eval 模式下 DetectionModel 返回 tuple/list，实际预测张量在 [0]，形状 [B, 4+nc, N]
        p0 = preds[0] if isinstance(preds, (list, tuple)) else preds
        # NMS 前候选数：与 ops.py 一致 xc = prediction[:, 4:4+nc].amax(1) > conf_thres
        # p0 形状 [B, 4+nc, N] -> 按「通道」切片（行），再对通道取 max
        pre_nms = [int((p0[i][4:4 + nc, :].amax(0) > args.conf).sum())
                   for i in range(p0.shape[0])]
        out = non_max_suppression(preds, args.conf, args.iou, nc=nc,
                                  max_det=MAX_DET, multi_label=True)
        for j, (i, p) in enumerate(sel):
            stem = Path(p).stem
            det = out[j]
            post_nms = 0 if det is None else len(det)
            if det is not None and len(det):
                det = det.clone()
                det[:, :4] = scale_boxes(x.shape[2:], det[:, :4], orig_shapes[j])
            post_trunc = min(post_nms, MAX_BOXES)
            orig_h, orig_w = orig_shapes[j]
            lb_img = letterbox(image=sub_imgs[j])
            top = (lb_img.shape[0] - sub_imgs[j].shape[0]) if lb_img.shape[0] > sub_imgs[j].shape[0] else 0
            rec = {
                "rgb_file": Path(p).name,
                "depth_file": Path(p).name,
                "orig_hw": [int(orig_h), int(orig_w)],
                "preprocessed_hw": [int(lb_img.shape[0]), int(lb_img.shape[1])],
                "pre_nms_candidates": pre_nms[j],
                "post_nms": post_nms,
                "post_max_boxes": post_trunc,
                "truncated": bool(post_nms > MAX_BOXES),
                "boxes": [],
            }
            if det is not None and len(det):
                d = det[np.argsort(-det[:, 4].numpy())][:20]
                for row in d:
                    rec["boxes"].append({
                        "x1": round(float(row[0]), 3), "y1": round(float(row[1]), 3),
                        "x2": round(float(row[2]), 3), "y2": round(float(row[3]), 3),
                        "conf": round(float(row[4]), 6), "cls": int(round(float(row[5]))),
                    })
            records[stem] = rec
        if len(records) >= len(picks):
            break

    json.dump({"env": env, "records": records}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[audit] 已写出 {args.out}：{len(records)} 张图")
    print(f"[audit] weights sha256 = {env['weights_sha256']}")
    print(f"[audit] 每图 pre_nms/post_nms/post_max_boxes:")
    for s in picks:
        r = records.get(s)
        if r:
            print(f"   {s:<24} {r['pre_nms_candidates']:>6} / {r['post_nms']:>5} / {r['post_max_boxes']:>4}"
                  f"   orig={r['orig_hw']} lb={r['preprocessed_hw']}")


if __name__ == "__main__":
    main()
