"""scripts/ensemble_final_candidate.py — RGBID:F4 = 3:1 最终候选融合（置信度合法版）。

== 与审计版 (inference_audit.py --task ensemble) 的唯一差别 ==
审计版把 3:1 权重直接乘进 conf 并写盘 → 产生 conf>1.0 的非法 TXT（已追溯：
inference_audit.py:192 `d_a[:, 4] *= w_ratio`，属融合公式副作用）。

本版把权重**只用于 NMS 胜者选择**（候选框融合的优先级），写盘时输出**该框的原始置信度**。
由于原始置信度来自 non_max_suppression 的 sigmoid 输出，恒有 conf ∈ (0, 1]，
**结构上不可能产生 conf>1**，无需截断、无需修复。

融合后的框集合与审计版 3:1 完全一致（仅写出的 conf 值不同），已由 verify_mode 校验。

未修改任何冻结文件（predict_rect.py / predict.py 均只读复用）。

用法:
  # val 对照（与 baseline 同一评价口径）
  python scripts/ensemble_final_candidate.py --source data/processed/rgbid_split/images/val/visible \
      --output diagnostic/rgbid_inference_gain/RGBID_f4_ens_3to1_legal --max_boxes 300
  # 测试集（1000 图，提交用 max_boxes=100，与冻结提交约定一致）
  python scripts/ensemble_final_candidate.py --source data/raw/test/visible \
      --output diagnostic/rgbid_inference_gain/TEST_ens_3to1 --max_boxes 100
  # 等价性校验：新框集合 vs 审计版框集合
  python scripts/ensemble_final_candidate.py --verify-only \
      --output diagnostic/rgbid_inference_gain/RGBID_f4_ens_3to1_legal \
      --ref diagnostic/rgbid_inference_gain/RGBID_f4_ens_3to1/results
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.loaders import LoadImagesAndVideos  # noqa: E402
from torchvision.ops import nms as torch_nms  # noqa: E402

from predict_rect import _preprocess_rect  # noqa: E402  (只读复用，未修改)
from predict import _to_chw, _format_lines, _load_yaml  # noqa: E402  (只读复用，未修改)

DEFAULT_W_A = "runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt"
DEFAULT_CFG_A = "configs/train_rgbird_ir_quicktest.yaml"
DEFAULT_W_B = "runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt"
DEFAULT_CFG_B = "configs/train_f4_1280.yaml"

WEIGHT_A, WEIGHT_B = 3.0, 1.0   # RGBID : F4 = 3 : 1


def _parse_args():
    ap = argparse.ArgumentParser(description="RGBID:F4=3:1 最终候选融合（conf 合法）")
    ap.add_argument("--weights_a", default=DEFAULT_W_A)
    ap.add_argument("--weights_b", default=DEFAULT_W_B)
    ap.add_argument("--train_config_a", default=DEFAULT_CFG_A)
    ap.add_argument("--train_config_b", default=DEFAULT_CFG_B)
    ap.add_argument("--source", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max_det", type=int, default=300)
    ap.add_argument("--max_boxes", type=int, default=100)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--ref", default=None, help="审计版 results 目录（等价性校验用）")
    return ap.parse_args()


def _load_cfg(path: Path):
    cfg = _load_yaml(path) if path.exists() else {}
    return (str(cfg.get("use_simotm", "RGBD")),
            list(cfg.get("pairs_rgb_ir", ["visible", "depth"])),
            list(cfg.get("pairs_rgb_depth", ["visible", "depth"])),
            int(cfg.get("channels", 4)))


def _load_model(weights: Path, device):
    model = YOLO(str(weights))
    nn = model.model.to(device).eval()
    nc = int(getattr(nn, "nc", 12))
    stride = int(nn.stride.max()) if getattr(nn, "stride", None) is not None else 32
    return nn, nc, stride


def _infer_rect(nn, im, imgsz, stride, args, nc, device):
    """单图 -> 原图坐标 [N,6]=xyxy,conf,cls（conf 为模型原始置信度）。"""
    from ultralytics.utils.ops import non_max_suppression, scale_boxes
    padded, ratio_pad, orig_hw, _ = _preprocess_rect(im, imgsz, stride)
    x = torch.from_numpy(_to_chw(padded)).unsqueeze(0).to(device).float() / 255.0
    with torch.no_grad():
        preds = nn(x)
    out = non_max_suppression(preds, args.conf, args.iou, nc=nc,
                              max_det=args.max_det, multi_label=True)[0]
    if out is not None and len(out):
        out[:, :4] = scale_boxes(x.shape[2:], out[:, :4], orig_hw, ratio_pad=ratio_pad)
    return out


def _nms_keep_indices(dets: torch.Tensor, scores: torch.Tensor, iou: float) -> torch.Tensor:
    """逐类贪心 NMS，返回保留行号（与 predict._nms_merge 的逐类 torch_nms 等价）。

    与冻结的 `_nms_merge` 的唯一区别：后者直接返回行本身（因此会带出加权后的 conf），
    这里返回索引，以便调用方取回**原始未加权**的行。
    """
    if len(dets) <= 1:
        return torch.arange(len(dets), dtype=torch.long)
    keep = []
    for c in dets[:, 5].unique():                 # 与 _nms_merge 相同的类遍历顺序
        m = dets[:, 5] == c
        idx = m.nonzero(as_tuple=True)[0]
        k = torch_nms(dets[idx, :4], scores[idx], iou)
        keep.append(idx[k])
    return torch.cat(keep) if keep else torch.zeros(0, dtype=torch.long)


def _fuse_3to1(d_a, d_b, iou):
    """3:1 候选框融合：权重仅决定 NMS 胜者，写盘保留原始 conf。"""
    parts, weights = [], []
    if d_a is not None and len(d_a):
        parts.append(d_a); weights.append(WEIGHT_A)
    if d_b is not None and len(d_b):
        parts.append(d_b); weights.append(WEIGHT_B)
    if not parts:
        return None
    dets = torch.cat(parts, dim=0)
    scores = dets[:, 4].clone()
    off = 0
    for p, w in zip(parts, weights):
        scores[off:off + len(p)] *= w          # 仅用于排序，不回写 dets
        off += len(p)
    keep = _nms_keep_indices(dets, scores, iou)
    return dets[keep]                          # 原始 conf，恒 ∈ (0,1]


def verify(args):
    """等价性校验：新（合法版）框集合与审计版权重版框集合是否一致。"""
    new_dir = (PROJECT_ROOT / args.output).resolve() / "results"
    ref_dir = (PROJECT_ROOT / args.ref).resolve()
    stems = sorted(p.stem for p in new_dir.glob("*.txt"))
    print(f"[verify] 新={new_dir}\n[verify] 参考={ref_dir}\n[verify] 图片数={len(stems)}")
    same_boxes = diff_boxes = 0
    n_over1_new = n_over1_ref = 0
    for stem in stems:
        def _rows(d):
            p = d / f"{stem}.txt"
            if not p.exists():
                return None
            return [ln.split() for ln in p.read_text().splitlines() if ln.strip()]
        a, b = _rows(new_dir), _rows(ref_dir)
        if a is None or b is None:
            diff_boxes += 1
            continue
        n_over1_new += sum(1 for r in a if len(r) == 6 and float(r[5]) > 1.0)
        n_over1_ref += sum(1 for r in b if len(r) == 6 and float(r[5]) > 1.0)
        key = lambda rows: sorted(tuple(round(float(v), 5) for v in r[1:5]) for r in rows)
        if key(a) == key(b):
            same_boxes += 1
        else:
            diff_boxes += 1
    print(f"[verify] 框集合完全一致的图片: {same_boxes}/{len(stems)}  (不一致 {diff_boxes})")
    print(f"[verify] conf>1 行数: 新版={n_over1_new}  审计版={n_over1_ref}")
    if diff_boxes == 0:
        print("[verify] ✅ 框集合逐图完全一致 → 权重改动只影响写出的 conf，不影响候选框")
    else:
        print("[verify] ⚠️ 存在不一致，需人工检查")


def main():
    args = _parse_args()
    if args.verify_only:
        verify(args)
        return
    if args.source is None:
        raise SystemExit("--source 必填（非 verify-only 模式）")

    t0 = time.time()
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    stm_a, ir_a, dep_a, _ = _load_cfg(PROJECT_ROOT / args.train_config_a)
    stm_b, ir_b, dep_b, _ = _load_cfg(PROJECT_ROOT / args.train_config_b)
    src = (PROJECT_ROOT / args.source).resolve()

    nn_a, nc_a, stride_a = _load_model((PROJECT_ROOT / args.weights_a).resolve(), device)
    nn_b, nc_b, stride_b = _load_model((PROJECT_ROOT / args.weights_b).resolve(), device)
    assert nc_a == nc_b == 12, f"类别数不一致: A={nc_a} B={nc_b}"

    loader_a = LoadImagesAndVideos(str(src), batch=1, use_simotm=stm_a, imgsz=args.imgsz,
                                   pairs_rgb_ir=ir_a, pairs_rgb_depth=dep_a)
    loader_b = LoadImagesAndVideos(str(src), batch=1, use_simotm=stm_b, imgsz=args.imgsz,
                                   pairs_rgb_ir=ir_b, pairs_rgb_depth=dep_b)
    n = loader_a.ni
    print(f"[final] source={src} n={n}")
    print(f"[final] A: {Path(args.weights_a).name} use_simotm={stm_a} ch=5 | "
          f"B: {Path(args.weights_b).name} use_simotm={stm_b} ch=4")
    print(f"[final] 融合 RGBID:F4 = {WEIGHT_A:.0f}:{WEIGHT_B:.0f} (权重仅用于 NMS 胜者选择)")
    print(f"[final] conf={args.conf} iou={args.iou} imgsz={args.imgsz} "
          f"max_det={args.max_det} max_boxes={args.max_boxes}")

    out_dir = (PROJECT_ROOT / args.output).resolve() / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    it_b = iter(loader_b)
    processed = 0
    for paths, imgs_a, _ in loader_a:
        _, imgs_b, _ = next(it_b)
        im_a, im_b = imgs_a[0], imgs_b[0]

        # 两模型各自独立推理，各自 ratio_pad 映射回同一原图坐标系
        d_a = _infer_rect(nn_a, im_a, args.imgsz, stride_a, args, nc_a, device)
        d_b = _infer_rect(nn_b, im_b, args.imgsz, stride_b, args, nc_b, device)
        dets = _fuse_3to1(d_a, d_b, args.iou)

        h, w = im_a.shape[:2]
        lines = _format_lines(dets, h, w, args.max_boxes)
        (out_dir / (Path(paths[0]).stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        processed += 1

    el = time.time() - t0
    print(f"[final] 完成 {processed}/{n} -> {out_dir}（{len(list(out_dir.glob('*.txt')))} 个 TXT）")
    print(f"[final] 总耗时 {el:.1f}s，平均 {el / max(processed, 1):.3f}s/图")


if __name__ == "__main__":
    main()
