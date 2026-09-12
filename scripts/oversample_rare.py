#!/usr/bin/env python3
# ============================================================
# 离线类别感知过采样（解决类别不平衡，非几何增强）
# ============================================================
# 背景：2026-09 消融证明【均匀 4× 几何增强有害】(0.312→0.215)——近重复样本
# 导致过拟合。但类别极度不平衡是确凿短板：tricycle 训练集仅 ~8 实例、ball ~76、
# boat ~112、uav ~120，在 BCE loss 里被 person(~4200)/animal(~2900) 彻底淹没。
#
# 本脚本做的是【过采样】，不是几何增强：
#   - 统计训练集各类别实例数 N_c；
#   - 对 N_c < target_count 的稀有类，其所在图按 ratio=ceil(target/N_c) 重复，
#     但封顶 max_repeat（tricycle 8→20× 封顶，无法靠复制追平 person，属数据上限）；
#   - 副本用【硬链接】(os.link) 生成，磁盘零额外占用（同 inode，同一挂载点）；
#   - 副本命名 `<stem>_bal<i>`，train.py 的 _split_by_group 会按 stem 分组，
#     原图 + _balN 副本进同一侧，杜绝近重复泄漏进 val；
#   - 副本字节级相同，多样性靠【在线 mosaic+hsv】提供——训练时每个副本被随机
#     mosaic/HSV 打散成不同视角。这正是与失败几何增强的关键区别：前者是固定
#     近重复、后者是每次随机的多样本（等价于"稀有类多训几遍 + 每次新增强"）。
#
# 用法（AutoDL）:
#   python scripts/oversample_rare.py \
#       --input_dir  /root/autodl-tmp/urban_multimodal_detection/data/raw/train \
#       --output_dir /root/autodl-tmp/urban_multimodal_detection/data/raw/train_bal \
#       --target_count 1000 --max_repeat 20
#
# 之后用 configs/dataset_bal.yaml 训练（train: train_bal/visible）。
# ============================================================

import argparse
import math
import os
import shutil
from collections import Counter
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_classes(label_path):
    """读取标注，返回该图出现的类别 id 集合。无标注/异常返回空集。"""
    classes = set()
    if not label_path.exists():
        return classes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    classes.add(int(float(parts[0])))
                except ValueError:
                    continue
    return classes


def count_instances(input_dir):
    """统计训练集各类别实例数 N_c，返回 Counter(cls -> count)。"""
    counter = Counter()
    lbl_dir = input_dir / "labels"
    if not lbl_dir.is_dir():
        lbl_dir = input_dir / "visible"  # 回退：标注紧邻图像
    for p in sorted(lbl_dir.glob("*.txt")):
        with open(p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        counter[int(float(parts[0]))] += 1
                    except ValueError:
                        continue
    return counter


def hardlink(src, dst):
    """硬链接（省磁盘），跨设备/不支持时回退拷贝。"""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(str(src), str(dst))


def main():
    ap = argparse.ArgumentParser(description="离线类别感知过采样（解决类别不平衡）")
    ap.add_argument("--input_dir", default="data/raw/train")
    ap.add_argument("--output_dir", default="data/raw/train_bal")
    ap.add_argument("--target_count", type=int, default=1000,
                    help="每类目标实例数；N_c 低于此值的类视为稀有类")
    ap.add_argument("--max_repeat", type=int, default=20,
                    help="单图重复次数上限（防止对极稀有类过度过拟合）")
    ap.add_argument("--nc", type=int, default=12)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    vis_dir = input_dir / "visible"
    if not vis_dir.is_dir():
        raise FileNotFoundError(f"未找到 visible 目录: {vis_dir}")

    # 各类别实例数 + 重复因子
    N = count_instances(input_dir)
    if not N:
        raise RuntimeError("未读取到任何标注，检查 input_dir/labels")

    repeat_c = {}
    for c in range(args.nc):
        n = N.get(c, 0)
        repeat_c[c] = 1 if n >= args.target_count else min(
            args.max_repeat, max(1, math.ceil(args.target_count / max(n, 1))))

    print("=" * 60)
    print(f"类别实例数 N_c 与重复因子（target={args.target_count}, max_repeat={args.max_repeat}）:")
    for c in range(args.nc):
        n = N.get(c, 0)
        mark = "  <稀有>" if n < args.target_count else ""
        print(f"  class {c:2d}: N={n:5d}  repeat={repeat_c[c]:2d}{mark}")
    print("=" * 60)

    for sub in ("visible", "infrared", "depth", "labels"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    stems = sorted({p.stem for p in vis_dir.iterdir() if p.suffix.lower() in IMG_EXTS})
    total_out = 0
    for stem in stems:
        # 找可见光扩展名（三模态同扩展名，已核实）
        vis_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            p = vis_dir / f"{stem}{ext}"
            if p.exists():
                vis_path = p
                break
        if vis_path is None:
            continue
        ext = vis_path.suffix.lower()
        ir_path = input_dir / "infrared" / f"{stem}{ext}"
        dep_path = input_dir / "depth" / f"{stem}{ext}"
        lbl_path = input_dir / "labels" / f"{stem}.txt"
        if not lbl_path.exists():
            lbl_path = input_dir / "visible" / f"{stem}.txt"

        # 重复因子 = 图中出现的最稀有类的 factor
        repeat = max((repeat_c[c] for c in read_classes(lbl_path)), default=1)

        # 原图 1 份
        for src, dst in (
            (vis_path, output_dir / "visible" / f"{stem}{ext}"),
            (ir_path, output_dir / "infrared" / f"{stem}{ext}"),
            (dep_path, output_dir / "depth" / f"{stem}{ext}"),
            (lbl_path, output_dir / "labels" / f"{stem}.txt"),
        ):
            if src.exists():
                hardlink(src, dst)

        # 额外 repeat-1 份副本
        for i in range(1, repeat):
            for src, dst in (
                (vis_path, output_dir / "visible" / f"{stem}_bal{i}{ext}"),
                (ir_path, output_dir / "infrared" / f"{stem}_bal{i}{ext}"),
                (dep_path, output_dir / "depth" / f"{stem}_bal{i}{ext}"),
                (lbl_path, output_dir / "labels" / f"{stem}_bal{i}.txt"),
            ):
                if src.exists():
                    hardlink(src, dst)
        total_out += repeat

    print(f"\n原图 {len(stems)} 张 → 过采样后 {total_out} 张（{total_out / max(len(stems), 1):.2f}×）")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
