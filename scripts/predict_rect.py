"""scripts/predict_rect.py — F4 提交 pipeline 的 rect letterbox 修复版（A/B + 提交生成）。

背景（审计结论）：
  model.val() 通过 model.py:598 强制 rect=True，把每张图 letterbox 到矩形 736×1312
  （stride=32 对齐的最小矩形画布），并显式把真实 ratio_pad 传给 scale_boxes；
  而 scripts/predict.py 用 square 1280×1280，且 letterbox(image=im) 丢弃 ratio_pad、
  scale_boxes 自行重算 gain/pad。二者导致预测框数/置信度系统性偏移（val 0.53273
  vs 提交 ≈0.51863，−0.0141）。

本脚本复刻 model.val() 的推理预处理/后处理，其余（conf/iou/max_det/通道/写盘格式）
完全复用 predict.py，保证 A/B 只差在「rect vs square + 是否显式 ratio_pad」。

核心（最小侵入，不碰 predict.py / F4 权重 / 训练配置）：
  原图(4ch [B,G,R,D])
    → 长边缩放到 imgsz=1280（load_image rect_mode=True：ceil，无 scaleup 封顶）
    → 按 stride=32 算最小矩形画布（set_rectangle 公式）
    → rect padding（RGB+Depth 合并为单张 4ch，天然共享 ratio/pad）
    → forward + NMS
    → scale_boxes(..., ratio_pad=真实值)

用法:
  # A/B：square（当前 predict.py 逻辑）与 rect（修复）各跑一遍
  python scripts/predict_rect.py --source data/processed/depth_split_train/images/val/visible \
      --mode square --output diagnostic/rect_pipeline_fix/validation_square_predictions
  python scripts/predict_rect.py --source data/processed/depth_split_train/images/val/visible \
      --mode rect   --output diagnostic/rect_pipeline_fix/validation_rect_predictions
  # 提交（测试集）
  python scripts/predict_rect.py --source data/raw/test/visible --mode rect --max_boxes 100 \
      --output diagnostic/rect_pipeline_fix/submission_rect_fixed
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.data.loaders import LoadImagesAndVideos  # noqa: E402
from ultralytics.utils.ops import non_max_suppression, scale_boxes  # noqa: E402

# 复用 predict.py 的输出格式化/通道重排，保证 TXT 与提交格式逐字节一致
from predict import _to_chw, _format_lines, _load_yaml, _parse_pairs  # noqa: E402

DEFAULT_WEIGHTS = "runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt"
DEFAULT_TRAIN_CONFIG = "configs/train_f4_1280.yaml"


def _compute_rect_shape(h0: int, w0: int, imgsz: int, stride: int, pad: float = 0.5):
    """复刻 base.py set_rectangle() 的单图矩形画布尺寸（批内宽高比一致时等价）。

    batch_shapes = ceil(shape * imgsz / stride + pad) * stride
    宽高比 <1（横向）: shape=[ar, 1]；宽高比 >1（纵向）: shape=[1, 1/ar]。
    返回 (rect_h, rect_w)。
    """
    ar = h0 / w0
    shape = [ar, 1.0] if ar < 1 else [1.0, 1.0 / ar]
    rect_h = int(np.ceil(shape[0] * imgsz / stride + pad)) * stride
    rect_w = int(np.ceil(shape[1] * imgsz / stride + pad)) * stride
    return rect_h, rect_w


def _preprocess_rect(im: np.ndarray, imgsz: int, stride: int):
    """复刻 model.val() 的两步预处理：load_image(rect_mode=True) + rect LetterBox。

    Args:
        im: 4ch uint8 [B,G,R,D]（native，未 resize）。
    Returns:
        (padded_im, ratio_pad, orig_hw, resized_hw)
        ratio_pad = ((ratio, ratio), (left, top))，ratio 为 load_image 缩放比
        （letterbox 因 rect 画布 >= 内容而 r=1.0，不再缩放，故总 ratio == load_image ratio）。
    """
    h0, w0 = im.shape[:2]

    # 1) load_image(rect_mode=True)：长边 -> imgsz，ceil，无 scaleup 封顶（小图放大 2×）
    r = imgsz / max(h0, w0)
    w = min(math.ceil(w0 * r), imgsz)
    h = min(math.ceil(h0 * r), imgsz)
    if (h, w) != (h0, w0):
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    resized_hw = (h, w)

    # 2) rect LetterBox（scaleup=False, center=True, stride 对齐）
    rect_h, rect_w = _compute_rect_shape(h0, w0, imgsz, stride)
    ratio = min(rect_h / h, rect_w / w)
    ratio = min(ratio, 1.0)  # scaleup=False
    new_unpad = (int(round(w * ratio)), int(round(h * ratio)))
    dw = (rect_w - new_unpad[0]) / 2.0
    dh = (rect_h - new_unpad[1]) / 2.0
    top = int(round(dh - 0.1))
    left = int(round(dw - 0.1))
    bottom = rect_h - new_unpad[1] - top
    right = rect_w - new_unpad[0] - left
    padded = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114, 114)
    )
    ratio_pad = ((r, r), (left, top))
    return padded, ratio_pad, (h0, w0), resized_hw


def _preprocess_square(im: np.ndarray, imgsz: int, stride: int, scaleup: bool):
    """复刻 scripts/predict.py 当前的 square letterbox（对照组）。

    predict.py 用 letterbox(image=im) 只拿 img、丢弃 ratio_pad，scale_boxes 再重算
    gain/pad（scaleup=True 下恰好与真实一致）。这里返回 (padded_im, ratio_pad, orig_hw)
    供 scale_boxes 使用；ratio_pad 由重算逻辑给出，与 predict.py:399 等价。
    """
    lb = LetterBox(new_shape=(imgsz, imgsz), auto=False, scaleFill=False,
                   scaleup=scaleup, center=True, stride=stride)
    h0, w0 = im.shape[:2]
    img = lb(image=im)
    # 复刻 scale_boxes(ratio_pad=None) 的重算逻辑（predict.py:399 的实际行为）
    gain = min(imgsz / h0, imgsz / w0)
    pad = (round((imgsz - w0 * gain) / 2 - 0.1), round((imgsz - h0 * gain) / 2 - 0.1))
    return img, ((gain, gain), pad), (h0, w0), im.shape[:2]


def _parse_args():
    ap = argparse.ArgumentParser(description="F4 rect letterbox 修复版推理（A/B + 提交）")
    ap.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    ap.add_argument("--source", type=str,
                    default="data/processed/depth_split_train/images/val/visible",
                    help="可见光主输入目录（val 或 test）")
    ap.add_argument("--train_config", type=str, default=DEFAULT_TRAIN_CONFIG)
    ap.add_argument("--use_simotm", type=str, default=None)
    ap.add_argument("--pairs_rgb_ir", type=str, default=None)
    ap.add_argument("--pairs_rgb_depth", type=str, default=None)
    ap.add_argument("--channels", type=int, default=None)
    ap.add_argument("--mode", type=str, default="rect", choices=["rect", "square"],
                    help="rect=修复后；square=复刻当前 predict.py")
    ap.add_argument("--scaleup", type=str, default="true", choices=["true", "false"],
                    help="仅 square 模式生效（复刻 predict.py 的 scaleup 参数）")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--max_boxes", type=int, default=300,
                    help="每图写入最大框数；A/B 用 300（对齐 val 不截断），提交用 100")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--output", type=str, required=True, help="结果目录（写 <output>/results/*.txt）")
    return ap.parse_args()


def main():
    args = _parse_args()

    weights = (PROJECT_ROOT / args.weights).resolve()
    source_dir = (PROJECT_ROOT / args.source).resolve()
    train_cfg = _load_yaml(PROJECT_ROOT / args.train_config) \
        if (PROJECT_ROOT / args.train_config).exists() else {}

    use_simotm = args.use_simotm or str(train_cfg.get("use_simotm", "RGBD"))
    pairs_rgb_ir = (_parse_pairs(args.pairs_rgb_ir) if args.pairs_rgb_ir is not None
                    else list(train_cfg.get("pairs_rgb_ir", ["visible", "depth"])))
    pairs_rgb_depth = (_parse_pairs(args.pairs_rgb_depth) if args.pairs_rgb_depth is not None
                       else list(train_cfg.get("pairs_rgb_depth", ["visible", "depth"])))
    channels = int(args.channels if args.channels is not None else train_cfg.get("channels", 4))
    # IR 对比度处理（RGBID 专用）：与训练侧 base.py 同源。旧配置无此键 -> percentile =
    # 既有行为，冻结提交链输出逐位不变；D(CLAHE) 由 configs/train_rgbid_ir_clahe.yaml 携带。
    ir_encoding = str(train_cfg.get("ir_encoding", "percentile") or "percentile")

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")

    model = YOLO(str(weights))
    nn = model.model.to(device).eval()
    nc = int(getattr(nn, "nc", 12))
    stride = int(nn.stride.max()) if getattr(nn, "stride", None) is not None else 32

    out_root = (PROJECT_ROOT / args.output).resolve()
    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    loader = LoadImagesAndVideos(
        str(source_dir), batch=args.batch, use_simotm=use_simotm,
        imgsz=args.imgsz, pairs_rgb_ir=pairs_rgb_ir, pairs_rgb_depth=pairs_rgb_depth,
        ir_encoding=ir_encoding,
    )
    n_test = loader.ni
    scaleup = (args.scaleup == "true")

    print(f"[predict_rect] mode={args.mode} scaleup={scaleup} weights={weights.name}")
    print(f"[predict_rect] source={source_dir} n={n_test} conf={args.conf} iou={args.iou} "
          f"imgsz={args.imgsz} max_det={args.max_det} max_boxes={args.max_boxes}")
    print(f"[predict_rect] use_simotm={use_simotm} channels={channels} ir_encoding={ir_encoding}")

    processed = 0
    for paths, imgs, _info in loader:
        # 预处理每张图
        pre = []
        for im in imgs:
            if args.mode == "rect":
                pre.append(_preprocess_rect(im, args.imgsz, stride))
            else:
                pre.append(_preprocess_square(im, args.imgsz, stride, scaleup))

        # 按画布形状分组（同宽高比 → 同 rect 形状 → 可 stack 批推理）
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
                    ratio_pad = pre[j][1]
                    orig_hw = pre[j][2]
                    if args.mode == "rect":
                        o[:, :4] = scale_boxes(x.shape[2:], o[:, :4], orig_hw, ratio_pad=ratio_pad)
                    else:
                        # 复刻 predict.py:399（不传 ratio_pad，scale_boxes 自行重算）
                        o[:, :4] = scale_boxes(x.shape[2:], o[:, :4], orig_hw)
                outputs[j] = o

        for i, p in enumerate(paths):
            orig_h, orig_w = pre[i][2]
            lines = _format_lines(outputs[i], orig_h, orig_w, args.max_boxes)
            (results_dir / (Path(p).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            processed += 1

    n_txt = len(list(results_dir.glob("*.txt")))
    print(f"[predict_rect] 完成 {processed} 张图 → {results_dir}（{n_txt} 个 TXT）")


if __name__ == "__main__":
    main()
