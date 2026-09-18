"""scripts/modality_ablation_infer.py — 模态消融推理（只读分析，不训练、不提交）。

用途：评估训练好的 RGBID 模型对 IR / Depth 两个模态的依赖程度。
在推理输入端**整体替换**某一模态通道，其余管线与冻结提交管线**逐字节一致**。

  --drop none   : RGB + IR          + Depth           （= 常规推理，对照组）
  --drop ir     : RGB + IR(=125)    + Depth           （IR 通道整体置 125）
  --drop depth  : RGB + IR          + Depth(=0)       （Depth 通道整体置 0）

填充值与训练期 Modality Dropout 完全一致（ir_fill=125 / depth_fill=0），
且**不改变 shape**（恒为 [B,5,H,W]）。

⚠️ 本脚本产物**仅用于模态依赖分析**，不得用于正式比赛提交。

管线复用（全部 import，不复制、不修改）：
  - `predict_rect._preprocess_rect`  ：rect letterbox（与 model.val() 一致）
  - `predict._to_chw`               ：[B,G,R,IR,D] → [R,G,B,IR,D]
  - `predict._format_lines`         ：归一化 + clamp + 6 位小数写盘

通道索引依据 `base.py::_merge_channels_rgbid`：`cv2.merge((b,g,r,ir,d))` → 索引 3=IR，4=Depth。

用法:
  python scripts/modality_ablation_infer.py \
      --weights runs/urban_multimodal_det_yolo11_rgbid_modality_dropout/weights/best.pt \
      --train_config configs/train_rgbid_modality_dropout.yaml \
      --source data/processed/rgbid_split/images/val/visible \
      --drop ir --output diagnostic/modality_dropout/RGBID_md_ir_dropped
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _p in (str(PROJECT_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ultralytics import YOLO                                    # noqa: E402
from ultralytics.data.loaders import LoadImagesAndVideos        # noqa: E402
from ultralytics.utils.ops import non_max_suppression, scale_boxes  # noqa: E402

from predict import _to_chw, _format_lines, _load_yaml, _parse_pairs  # noqa: E402
from predict_rect import _preprocess_rect                       # noqa: E402

# 与 configs/train_rgbid_modality_dropout.yaml 的 modality_dropout 保持一致
FILL = {"ir": (3, 125), "depth": (4, 0)}


def _parse_args():
    ap = argparse.ArgumentParser(description="RGBID 模态消融推理（分析用，非提交）")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--source", type=str,
                    default="data/processed/rgbid_split/images/val/visible")
    ap.add_argument("--train_config", type=str,
                    default="configs/train_rgbid_modality_dropout.yaml")
    ap.add_argument("--drop", type=str, default="none", choices=["none", "ir", "depth"],
                    help="要整体替换的模态通道")
    ap.add_argument("--use_simotm", type=str, default=None)
    ap.add_argument("--pairs_rgb_ir", type=str, default=None)
    ap.add_argument("--pairs_rgb_depth", type=str, default=None)
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--max_boxes", type=int, default=300)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--output", type=str, required=True)
    return ap.parse_args()


def apply_modality_fill(im: np.ndarray, drop: str) -> np.ndarray:
    """把指定模态通道整体替换为填充值；shape/dtype 不变。"""
    if drop == "none":
        return im
    if im.ndim != 3 or im.shape[2] != 5:
        raise ValueError(f"--drop {drop} 需要 5ch [B,G,R,IR,D] 输入，收到 {im.shape}")
    ch, val = FILL[drop]
    out = im.copy()
    out[:, :, ch] = val          # 不就地修改，避免污染 loader 缓冲
    return out


def main():
    args = _parse_args()

    weights = (PROJECT_ROOT / args.weights).resolve()
    source_dir = (PROJECT_ROOT / args.source).resolve()
    train_cfg = _load_yaml(PROJECT_ROOT / args.train_config) \
        if (PROJECT_ROOT / args.train_config).exists() else {}

    use_simotm = args.use_simotm or str(train_cfg.get("use_simotm", "RGBID"))
    pairs_rgb_ir = (_parse_pairs(args.pairs_rgb_ir) if args.pairs_rgb_ir is not None
                    else list(train_cfg.get("pairs_rgb_ir", ["visible", "infrared"])))
    pairs_rgb_depth = (_parse_pairs(args.pairs_rgb_depth) if args.pairs_rgb_depth is not None
                       else list(train_cfg.get("pairs_rgb_depth", ["visible", "depth"])))
    channels = int(args.channels if args.channels is not None else train_cfg.get("channels", 5))

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")

    model = YOLO(str(weights))
    nn = model.model.to(device).eval()
    nc = int(getattr(nn, "nc", 12))
    stride = int(nn.stride.max()) if getattr(nn, "stride", None) is not None else 32

    results_dir = (PROJECT_ROOT / args.output).resolve() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    loader = LoadImagesAndVideos(
        str(source_dir), batch=args.batch, use_simotm=use_simotm,
        imgsz=args.imgsz, pairs_rgb_ir=pairs_rgb_ir, pairs_rgb_depth=pairs_rgb_depth,
    )

    print(f"[modality_ablation] drop={args.drop} weights={weights.name} channels={channels}")
    print(f"[modality_ablation] source={source_dir} n={loader.ni} conf={args.conf} "
          f"iou={args.iou} imgsz={args.imgsz} max_det={args.max_det} max_boxes={args.max_boxes}")
    if args.drop != "none":
        ch, val = FILL[args.drop]
        print(f"[modality_ablation] 通道 {ch} 整体替换为 {val}")

    processed = 0
    for paths, imgs, _info in loader:
        pre = []
        for im in imgs:
            im = apply_modality_fill(im, args.drop)
            pre.append(_preprocess_rect(im, args.imgsz, stride))

        groups: dict = {}
        for i, (padded, *_rest) in enumerate(pre):
            groups.setdefault(padded.shape[:2], []).append(i)

        outputs = [None] * len(imgs)
        for shape, idxs in groups.items():
            x = torch.stack([torch.from_numpy(_to_chw(pre[j][0])) for j in idxs]).to(device).float() / 255.0
            with torch.no_grad():
                preds = nn(x)
            out = non_max_suppression(preds, args.conf, args.iou, nc=nc,
                                      max_det=args.max_det, multi_label=True)
            for k, j in enumerate(idxs):
                o = out[k]
                if o is not None and len(o):
                    o[:, :4] = scale_boxes(x.shape[2:], o[:, :4],
                                           pre[j][2], ratio_pad=pre[j][1])
                outputs[j] = o

        for i, p in enumerate(paths):
            orig_h, orig_w = pre[i][2]
            lines = _format_lines(outputs[i], orig_h, orig_w, args.max_boxes)
            (results_dir / (Path(p).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            processed += 1

    n_txt = len(list(results_dir.glob("*.txt")))
    print(f"[modality_ablation] 完成 {processed} 张图 → {results_dir}（{n_txt} 个 TXT）")


if __name__ == "__main__":
    main()
