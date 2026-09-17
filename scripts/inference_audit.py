"""scripts/inference_audit.py — RGBID 单模型 TTA / RGBID+F4 Ensemble 推理增益审计。

本脚本只做「推理」，不改模型结构、不改 predict_rect.py、不改评价代码、不改 NMS 默认参数。

设计原则（与冻结 pipeline 严格对齐）:
  - 复用冻结的 ``predict_rect.py::_preprocess_rect`` 作为唯一预处理入口（rect letterbox，
    与 model.val() 一致，也是线上提交 pipeline 的预处理）。
  - 复用 ``predict.py`` 的 ``_to_chw`` / ``_nms_merge`` / ``_format_lines``，保证通道重排、
    逐类 NMS、TXT 写盘与 baseline（predict_rect.py --mode rect）逐字节一致。
  - NMS 参数固定 conf=0.001 / iou=0.7 / max_det=300（与正式评估一致，不得改）。

任务:
  hflip      — 原图 + 水平翻转（5ch 整体翻转，RGB/IR/D 同步翻转），逐类 NMS 合并。
  multiscale — 0.8x / 1.0x / 1.2x 三尺度 rect 推理，逐类 NMS 合并。
  ensemble   — RGBID(5ch) + F4(4ch) 独立推理 → 加权 conf 拼接 → 逐类 NMS 合并。

用法:
  python scripts/inference_audit.py --task hflip
  python scripts/inference_audit.py --task multiscale
  python scripts/inference_audit.py --task ensemble --ratio 2:1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.loaders import LoadImagesAndVideos  # noqa: E402
from ultralytics.utils.ops import non_max_suppression, scale_boxes  # noqa: E402

# 复用冻结 pipeline 的预处理 / 通道重排 / NMS 合并 / 写盘
from predict_rect import _preprocess_rect  # noqa: E402
from predict import _to_chw, _format_lines, _nms_merge, _load_yaml, _parse_pairs  # noqa: E402

DEFAULT_WEIGHTS_A = "runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt"
DEFAULT_CFG_A = "configs/train_rgbird_ir_quicktest.yaml"
DEFAULT_WEIGHTS_B = "runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt"
DEFAULT_CFG_B = "configs/train_f4_1280.yaml"
DEFAULT_SOURCE = "data/processed/rgbid_split/images/val/visible"


def _parse_args():
    ap = argparse.ArgumentParser(description="RGBID TTA / Ensemble 推理增益审计")
    ap.add_argument("--task", required=True, choices=["hflip", "multiscale", "ensemble"])
    ap.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_A, help="模型 A (RGBID)")
    ap.add_argument("--weights_b", type=str, default=DEFAULT_WEIGHTS_B, help="模型 B (F4，仅 ensemble)")
    ap.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="可见光主输入目录")
    ap.add_argument("--train_config", type=str, default=DEFAULT_CFG_A, help="模型 A 训练配置")
    ap.add_argument("--train_config_b", type=str, default=DEFAULT_CFG_B, help="模型 B 训练配置")
    ap.add_argument("--ratio", type=str, default="1:1", help="ensemble 权重 RGBID:F4")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--max_boxes", type=int, default=300)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output", type=str, required=True)
    return ap.parse_args()


def _load_model(weights: Path, device: torch.device):
    model = YOLO(str(weights))
    nn = model.model.to(device).eval()
    nc = int(getattr(nn, "nc", 12))
    stride = int(nn.stride.max()) if getattr(nn, "stride", None) is not None else 32
    return nn, nc, stride


def _loader_params(cfg_path: Path):
    """从训练配置读多模态参数（与 predict_rect.py / evaluate.py 同一套字段）。"""
    cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}
    use_simotm = str(cfg.get("use_simotm", "RGBD"))
    pairs_rgb_ir = list(cfg.get("pairs_rgb_ir", ["visible", "depth"]))
    pairs_rgb_depth = list(cfg.get("pairs_rgb_depth", ["visible", "depth"]))
    channels = int(cfg.get("channels", 4))
    return use_simotm, pairs_rgb_ir, pairs_rgb_depth, channels


def _forward_rect(nn, im, imgsz, stride, conf, iou, max_det, nc, device):
    """单张 HWC uint8 多模态图 -> 原图坐标 [N,6]=xyxy,conf,cls（或 None）。"""
    padded, ratio_pad, orig_hw, _ = _preprocess_rect(im, imgsz, stride)
    x = torch.from_numpy(_to_chw(padded)).unsqueeze(0).to(device).float() / 255.0
    with torch.no_grad():
        preds = nn(x)
    out = non_max_suppression(preds, conf, iou, nc=nc, max_det=max_det, multi_label=True)[0]
    if out is not None and len(out):
        out[:, :4] = scale_boxes(x.shape[2:], out[:, :4], orig_hw, ratio_pad=ratio_pad)
    return out


def _merge_dets(dets_list, iou):
    """把多个 [N,6] 张量拼接后逐类 NMS（空则返回 None）。"""
    valid = [d for d in dets_list if d is not None and len(d)]
    if not valid:
        return None
    merged = torch.cat(valid, dim=0)
    return _nms_merge(merged, iou)


def main():
    args = _parse_args()
    t_start = time.time()

    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    use_simotm, pairs_ir, pairs_depth, channels = _loader_params(PROJECT_ROOT / args.train_config)
    source_dir = (PROJECT_ROOT / args.source).resolve()

    # 模型 A (RGBID)
    nn_a, nc_a, stride_a = _load_model((PROJECT_ROOT / args.weights).resolve(), device)
    print(f"[audit] task={args.task} ratio={args.ratio} device={device}")
    print(f"[audit] model A: {Path(args.weights)} nc={nc_a} stride={stride_a} "
          f"use_simotm={use_simotm} channels={channels}")

    loader_a = LoadImagesAndVideos(
        str(source_dir), batch=1, use_simotm=use_simotm,
        imgsz=args.imgsz, pairs_rgb_ir=pairs_ir, pairs_rgb_depth=pairs_depth,
    )
    n_test = loader_a.ni

    # 模型 B (F4, 仅 ensemble)
    nn_b = nc_b = stride_b = None
    loader_b = None
    w_ratio = 1.0
    if args.task == "ensemble":
        use_b, ir_b, dep_b, ch_b = _loader_params(PROJECT_ROOT / args.train_config_b)
        nn_b, nc_b, stride_b = _load_model((PROJECT_ROOT / args.weights_b).resolve(), device)
        print(f"[audit] model B: {Path(args.weights_b)} nc={nc_b} stride={stride_b} "
              f"use_simotm={use_b} channels={ch_b}")
        loader_b = LoadImagesAndVideos(
            str(source_dir), batch=1, use_simotm=use_b,
            imgsz=args.imgsz, pairs_rgb_ir=ir_b, pairs_rgb_depth=dep_b,
        )
        a_n, b_n = [int(x) for x in args.ratio.split(":")]
        w_ratio = a_n / b_n  # 模型 A(主) 相对 B 的权重倍率

    out_root = (PROJECT_ROOT / args.output).resolve()
    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    iter_b = iter(loader_b) if loader_b is not None else None
    for paths, imgs, _info in loader_a:
        p = Path(paths[0])
        im_a = imgs[0]  # 5ch [B,G,R,IR,D]

        dets = None
        if args.task == "hflip":
            d_orig = _forward_rect(nn_a, im_a, args.imgsz, stride_a,
                                   args.conf, args.iou, args.max_det, nc_a, device)
            im_flip = cv2.flip(im_a, 1)  # 5ch 整体水平翻转（RGB/IR/D 同步）
            d_flip = _forward_rect(nn_a, im_flip, args.imgsz, stride_a,
                                   args.conf, args.iou, args.max_det, nc_a, device)
            if d_flip is not None and len(d_flip):
                W = im_a.shape[1]
                d_flip = d_flip.clone()
                x1 = W - d_flip[:, 2]
                x2 = W - d_flip[:, 0]
                d_flip[:, 0] = x1
                d_flip[:, 2] = x2
            dets = _merge_dets([d_orig, d_flip], args.iou)

        elif args.task == "multiscale":
            all_dets = []
            for s in (0.8, 1.0, 1.2):
                sz = int(round(args.imgsz * s))
                d = _forward_rect(nn_a, im_a, sz, stride_a,
                                  args.conf, args.iou, args.max_det, nc_a, device)
                if d is not None and len(d):
                    all_dets.append(d)
            dets = _merge_dets(all_dets, args.iou)

        elif args.task == "ensemble":
            _, imgs_b, _ = next(iter_b)
            im_b = imgs_b[0]  # 4ch [B,G,R,D]
            d_a = _forward_rect(nn_a, im_a, args.imgsz, stride_a,
                                args.conf, args.iou, args.max_det, nc_a, device)
            d_b = _forward_rect(nn_b, im_b, args.imgsz, stride_b,
                                args.conf, args.iou, args.max_det, nc_b, device)
            if d_a is not None and len(d_a):
                d_a = d_a.clone()
                d_a[:, 4] *= w_ratio  # 主模型 A 加权
            if d_b is not None and len(d_b):
                d_b = d_b.clone()
                d_b[:, 4] *= 1.0     # B 基准权重
            dets = _merge_dets([d_a, d_b], args.iou)

        orig_h, orig_w = im_a.shape[:2]
        lines = _format_lines(dets, orig_h, orig_w, args.max_boxes)
        (results_dir / (p.stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        processed += 1

    elapsed = time.time() - t_start
    n_txt = len(list(results_dir.glob("*.txt")))
    print(f"[audit] 完成 {processed}/{n_test} 张图 -> {results_dir}（{n_txt} 个 TXT）")
    print(f"[audit] 总耗时 {elapsed:.1f}s，平均 {elapsed / max(processed, 1):.3f}s/图")


if __name__ == "__main__":
    main()
