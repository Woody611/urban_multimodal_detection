#!/usr/bin/env python3
# ============================================================
# 离线三模态数据增强脚本（仅几何变换，不做颜色增强）
# ============================================================
# 对训练集的三模态图像（可见光 visible / 红外 infrared / 深度 depth）做同步几何增强，
# 每张原图生成 num_aug 个增强变体（默认 3），并把【原图一并复制】进输出目录，
# 得到 原图 + 3 变体 = 4 倍数据，用于最终冲刺训练。
#
# 关键设计（与 fork 的加载约定完全对齐）:
#   - 同一张图的三模态使用【完全相同的几何变换矩阵】，保证空间对齐。
#   - 标注框（YOLO 归一化格式 cls cx cy w h）用同一矩阵同步变换：
#     先水平翻转 cx->1-cx，再对 4 个角点做仿射变换后重算外接框。
#   - 几何变换：水平翻转 p=0.5；旋转 ±5°；缩放 0.8~1.2；平移 ±5%（像素）。
#   - 深度图用最近邻插值 + 0 填充，保证无效像素（<300mm）保持 0、不参与插值混合，
#     且【保留原始格式】(16-bit PNG 毫米值 / 8-bit JPG)，训练加载器据此做毫米归一化。
#   - visible / infrared 用双线性插值 + 114 填充。
#   - 原图用字节级复制（shutil.copy2），不做重编码、无质量损失。
#   - 输出目录结构与 data/raw/train 一致: train_aug/{visible, infrared, depth, labels}，
#     train.py 的 _split_train_val_depth/rgbid 会读 src_dir.parent/"labels" 找到标注，
#     靠 file_path.replace("visible","depth"/"infrared") 找到第二/第三模态。
#   - 多进程 (ProcessPoolExecutor) + tqdm 进度条 + 断点续跑（已存在输出则跳过）。
#
# 关于「红外/深度复制成 3 通道」：本 fork 的加载器在读取时已做
#   np.stack([im,im,im], axis=-1) 复制成 3 通道（base.py Depth/Infrared 分支）。
#   增强脚本因此【保留原始格式】（深度保留 16-bit 毫米值），避免提前转 3 通道破坏
#   毫米归一化；加载器对「原图」和「增强图」的处理完全一致。
#
# 用法（AutoDL）:
#   python scripts/augment_offline.py \
#       --input_dir  /autodl-tmp/urban_multimodal_detection/data/raw/train \
#       --output_dir /autodl-tmp/urban_multimodal_detection/data/raw/train_aug \
#       --dataset_yaml /autodl-tmp/urban_multimodal_detection/data/processed/augmented_split/dataset.yaml \
#       --num_aug 3 --workers 8
# ============================================================

import argparse
import os
import random
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# 12 类（顺序即 class id，与 configs/dataset.yaml 保持一致）
CLASS_NAMES = [
    "person", "boat", "animal", "seat", "sign", "bicycle",
    "car", "ball", "light", "garbage_can", "uav", "tricycle",
]


# ------------------------------------------------------------------
# 仿射变换（旋转/缩放绕图像中心 + 平移），用 cv2.getRotationMatrix2D 保证正确性
# ------------------------------------------------------------------
def build_affine_matrix(H, W, angle_deg, scale, tx_px, ty_px):
    """返回 3x3 齐次仿射矩阵 M（源坐标 -> 目标坐标）。

    cv2.getRotationMatrix2D(center=(W/2,H/2), angle, scale) 给出绕图像中心的
    旋转+缩放矩阵（源->目标），再对 M[0,2]/M[1,2] 叠加平移即得最终仿射矩阵。
    图像用 cv2.warpAffine(img, M[:2])，点用 p_dest = M @ [x,y,1]^T，两者一致。
    """
    M = cv2.getRotationMatrix2D(center=(W / 2.0, H / 2.0), angle=angle_deg, scale=scale)
    M[0, 2] += tx_px
    M[1, 2] += ty_px
    return np.vstack([M, [0.0, 0.0, 1.0]])


def sample_transform(rng, H, W):
    """按增强规则随机采样: 返回 (hflip, M)。"""
    hflip = rng.random() < 0.5
    angle = rng.uniform(-5.0, 5.0)
    scale = rng.uniform(0.8, 1.2)
    tx = rng.uniform(-0.05, 0.05) * W
    ty = rng.uniform(-0.05, 0.05) * H
    return hflip, build_affine_matrix(H, W, angle, scale, tx, ty)


def warp_image(img, M, interp, border):
    """用仿射矩阵对图像做 warpAffine（保持原尺寸）。"""
    H, W = img.shape[:2]
    return cv2.warpAffine(img, M[:2], (W, H), flags=interp, borderValue=border)


def imwrite_opt(path, img):
    """写图：PNG 用最高压缩级别 9（无损，像素值逐位不变，8-bit 省 10~17%、16-bit 深度约 4%），JPG 正常写。"""
    if str(path).lower().endswith(".png"):
        return cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return cv2.imwrite(str(path), img)


# ------------------------------------------------------------------
# 标注框读写与变换
# ------------------------------------------------------------------
def read_yolo_labels(label_path):
    """读取 YOLO 标注，返回 (N,5) 归一化数组 [cls, cx, cy, w, h]。无标注返回空数组。"""
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            boxes.append([float(x) for x in parts[:5]])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 5)


def write_yolo_labels(label_path, boxes):
    """写回 YOLO 标注，坐标保留 6 位小数。"""
    with open(label_path, "w") as f:
        for b in boxes:
            f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")


def transform_boxes(boxes, hflip, M, H, W):
    """水平翻转（cx->1-cx）+ 仿射变换标注框（角点变换后重算外接框），裁剪到 [0,1]，过滤退化框。"""
    if len(boxes) == 0:
        return boxes
    cls = boxes[:, 0:1]
    cx, cy, w, h = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    if hflip:
        cx = 1.0 - cx
    # 4 个角点（像素坐标）
    x1 = (cx - w / 2) * W
    y1 = (cy - h / 2) * H
    x2 = (cx + w / 2) * W
    y2 = (cy + h / 2) * H
    corners = np.stack([x1, y1, x2, y1, x2, y2, x1, y2], axis=1).reshape(-1, 2)  # (4N,2)
    pts = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)      # (4N,3)
    pts = (M @ pts.T).T                                                          # 应用仿射变换
    pts = pts[:, :2].reshape(-1, 8)                                              # 每框 4 角点
    xs = pts[:, [0, 2, 4, 6]]
    ys = pts[:, [1, 3, 5, 7]]
    nx1, nx2 = xs.min(1), xs.max(1)
    ny1, ny2 = ys.min(1), ys.max(1)
    # 归一化并裁剪到 [0,1]
    nx1 = np.clip(nx1 / W, 0, 1)
    nx2 = np.clip(nx2 / W, 0, 1)
    ny1 = np.clip(ny1 / H, 0, 1)
    ny2 = np.clip(ny2 / H, 0, 1)
    ncx = (nx1 + nx2) / 2
    ncy = (ny1 + ny2) / 2
    nw = nx2 - nx1
    nh = ny2 - ny1
    out = np.concatenate([cls, ncx[:, None], ncy[:, None], nw[:, None], nh[:, None]], axis=1)
    # 过滤：宽高太小（退化）或中心完全出界
    keep = (nw > 1e-3) & (nh > 1e-3) & (ncx >= 0) & (ncx <= 1) & (ncy >= 0) & (ncy <= 1)
    return out[keep]


# ------------------------------------------------------------------
# 单张图的处理（多进程 worker）
# ------------------------------------------------------------------
def process_one(args):
    stem, input_dir, output_dir, num_aug, seed = args
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    for sub in ("visible", "infrared", "depth", "labels"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    # 三模态同名同扩展名（已核实：visible/infrared/depth 的 stem->ext 完全一致）
    vis_path = None
    for ext in (".jpg", ".png", ".jpeg"):
        p = input_dir / "visible" / f"{stem}{ext}"
        if p.exists():
            vis_path = p
            break
    if vis_path is None:
        return stem, "skip_missing", 0
    vis_ext = vis_path.suffix.lower()
    ir_path = input_dir / "infrared" / f"{stem}{vis_ext}"
    dep_path = input_dir / "depth" / f"{stem}{vis_ext}"

    # 标注：优先 data/raw/train/labels/（实际布局），回退到 visible/ 同目录
    lbl_dir = input_dir / "labels"
    lbl_path = lbl_dir / f"{stem}.txt"
    if not lbl_path.exists() and (input_dir / "visible" / f"{stem}.txt").exists():
        lbl_path = input_dir / "visible" / f"{stem}.txt"

    if not ir_path.exists() or not dep_path.exists():
        return stem, "skip_missing", 0

    dep_ext = dep_path.suffix.lower()

    # ---- 1) 复制原图（字节级，无重编码），实现「原图 + 3 变体 = 4 倍」 ----
    for src, dst in (
        (vis_path, output_dir / "visible" / f"{stem}{vis_ext}"),
        (ir_path, output_dir / "infrared" / f"{stem}{vis_ext}"),
        (dep_path, output_dir / "depth" / f"{stem}{dep_ext}"),
        (lbl_path, output_dir / "labels" / f"{stem}.txt"),
    ):
        if not dst.exists():
            shutil.copy2(str(src), str(dst))

    # ---- 2) 生成增强变体 ----
    vis = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)          # BGR uint8 3ch
    ir = cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)        # uint8 3ch（灰度堆叠）
    dep = cv2.imread(str(dep_path), cv2.IMREAD_UNCHANGED)      # uint16 1ch(毫米) 或 uint8 3ch
    if vis is None or ir is None or dep is None:
        return stem, "skip_unreadable", 0

    H, W = vis.shape[:2]
    boxes = read_yolo_labels(lbl_path)

    n_aug = 0
    for i in range(num_aug):
        out_stem = f"{stem}_aug{i}"
        out_vis = output_dir / "visible" / f"{out_stem}{vis_ext}"
        out_ir = output_dir / "infrared" / f"{out_stem}{vis_ext}"
        out_dep = output_dir / "depth" / f"{out_stem}{dep_ext}"
        out_lbl = output_dir / "labels" / f"{out_stem}.txt"

        # 断点续跑：所有输出都存在则跳过
        if out_vis.exists() and out_ir.exists() and out_dep.exists() and out_lbl.exists():
            continue

        rng = random.Random(seed * 100003 + i)                 # 确定性，便于复现
        hflip, M = sample_transform(rng, H, W)

        v, ir2, d = vis, ir, dep
        if hflip:                                              # 水平翻转（先翻转再仿射）
            v, ir2, d = np.fliplr(v), np.fliplr(ir2), np.fliplr(d)

        imwrite_opt(out_vis, warp_image(v, M, cv2.INTER_LINEAR, 114))
        imwrite_opt(out_ir, warp_image(ir2, M, cv2.INTER_LINEAR, 114))
        # 深度：最近邻插值 + 0 填充，保证无效像素(<300mm) 保持 0、不参与插值混合
        imwrite_opt(out_dep, warp_image(d, M, cv2.INTER_NEAREST, 0))
        write_yolo_labels(out_lbl, transform_boxes(boxes, hflip, M, H, W))
        n_aug += 1

    return stem, "ok", n_aug


# ------------------------------------------------------------------
# 生成 dataset.yaml（path 指向 data/raw，train 指向 train_aug/visible）
# ------------------------------------------------------------------
def write_dataset_yaml(yaml_path, data_root, train_subdir, val_subdir):
    lines = [
        f"path: {data_root}",
        f"train: {train_subdir}",
        f"val: {val_subdir}",
        "nc: 12",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def main():
    ap = argparse.ArgumentParser(description="离线三模态几何增强")
    ap.add_argument("--input_dir", default="/autodl-tmp/urban_multimodal_detection/data/raw/train")
    ap.add_argument("--output_dir", default="/autodl-tmp/urban_multimodal_detection/data/raw/train_aug")
    ap.add_argument("--dataset_yaml", default="/autodl-tmp/urban_multimodal_detection/data/processed/augmented_split/dataset.yaml")
    ap.add_argument("--num_aug", type=int, default=3, help="每张原图生成的增强变体数（默认 3，输出含原图共 4 倍）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_images", type=int, default=0, help=">0 时只处理前 N 张（调试用）")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    vis_dir = input_dir / "visible"
    if not vis_dir.exists():
        raise FileNotFoundError(f"未找到 visible 目录: {vis_dir}")

    # 输出目录结构（与 data/raw/train 一致）
    for sub in ("visible", "infrared", "depth", "labels"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    # 收集所有 visible 图片 stem（去扩展名）
    stems = sorted({p.stem for p in vis_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")})
    if args.max_images > 0:
        stems = stems[: args.max_images]

    workers = max(1, min(args.workers, os.cpu_count() or 1))
    tasks = [(s, str(input_dir), str(output_dir), args.num_aug, args.seed) for s in stems]

    t0 = time.time()
    ok = skip_missing = skip_unreadable = n_aug_total = 0
    print(f"[augment] 输入 {input_dir} 共 {len(stems)} 张，每张 {args.num_aug} 变体（含原图 4 倍），{workers} 进程")
    if workers == 1:
        results = [process_one(t) for t in tqdm(tasks, desc="augment")]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(tqdm(ex.map(process_one, tasks, chunksize=8), total=len(tasks), desc="augment"))

    for _, status, n in results:
        if status == "ok":
            ok += 1
            n_aug_total += n
        elif status == "skip_missing":
            skip_missing += 1
        else:
            skip_unreadable += 1

    # 写 dataset.yaml（train 指向增强目录，val 占位指向 test，训练时由 train.py 按 val_ratio 重切）
    data_root = str(input_dir.parent)          # train/ 的上一级 = data/raw
    write_dataset_yaml(Path(args.dataset_yaml), data_root, "train_aug/visible", "test/visible")

    # 统计输出文件数与大小
    def count(sub):
        d = output_dir / sub
        files = [f for f in d.iterdir() if f.is_file()]
        return len(files), sum(f.stat().st_size for f in files)

    n_vis, b_vis = count("visible")
    n_ir, b_ir = count("infrared")
    n_dep, b_dep = count("depth")
    n_lbl, b_lbl = count("labels")
    total_files = n_vis + n_ir + n_dep + n_lbl
    total_bytes = b_vis + b_ir + b_dep + b_lbl

    dt = time.time() - t0
    print("\n================ 完成 ================")
    print(f"原图数量          : {len(stems)}")
    print(f"成功处理          : {ok}（每张含原图 + {args.num_aug} 变体 = {args.num_aug + 1} 倍）")
    print(f"缺失/损坏跳过     : {skip_missing + skip_unreadable}")
    print(f"输出图片          : visible={n_vis}  infrared={n_ir}  depth={n_dep}")
    print(f"输出标注          : {n_lbl}")
    print(f"输出总文件数      : {total_files}")
    print(f"输出总大小        : {total_bytes / 1024 / 1024 / 1024:.2f} GiB")
    print(f"耗时              : {dt:.1f} s")
    print(f"dataset.yaml      : {args.dataset_yaml}")


if __name__ == "__main__":
    main()
