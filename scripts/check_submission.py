"""scripts/check_submission.py — 提交前自检（最终冲刺 · 任务一）。

对 predict_rect.py / predict.py 生成的 results/*.txt 做 10 点检查，并打包 submission.zip。
不改动任何预测脚本或权重；只读校验 + 打包。

用法:
  python scripts/check_submission.py --results submissions/rgbird_ir_quicktest/results \
      --zip submissions/rgbird_ir_quicktest/submission.zip --expect 1000
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np

CLASS_ID_MIN, CLASS_ID_MAX = 0, 11


def _parse_args():
    ap = argparse.ArgumentParser(description="提交前自检 + 打包 zip")
    ap.add_argument("--results", required=True, help="results 目录（含 *.txt）")
    ap.add_argument("--zip", required=True, help="输出 submission.zip 路径")
    ap.add_argument("--expect", type=int, default=1000, help="期望图片数")
    return ap.parse_args()


def main():
    args = _parse_args()
    results_dir = Path(args.results)
    txts = sorted(results_dir.glob("*.txt"))

    empty = []
    n_lines = 0
    bad_format = 0
    bad_class = 0
    bad_coord = 0
    boxes_per_img = []
    truncated = 0
    max_boxes_here = 0

    for txt in txts:
        lines = txt.read_text(encoding="utf-8").splitlines()
        lines = [ln for ln in lines if ln.strip()]
        if not lines:
            empty.append(txt.name)
            boxes_per_img.append(0)
            continue
        n_img = 0
        for ln in lines:
            n_lines += 1
            parts = ln.split()
            if len(parts) != 6:
                bad_format += 1
                continue
            try:
                cid = int(parts[0])
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                bad_format += 1
                continue
            if cid < CLASS_ID_MIN or cid > CLASS_ID_MAX:
                bad_class += 1
            if any(v < 0.0 or v > 1.0 for v in vals):
                bad_coord += 1
            n_img += 1
        boxes_per_img.append(n_img)
        max_boxes_here = max(max_boxes_here, n_img)
        if n_img >= 100:
            truncated += 1

    boxes = np.array(boxes_per_img)
    n = len(txts)
    print("=" * 64)
    print("提交前自检结果")
    print("=" * 64)
    print(f"[1] TXT 数量        : {n} / 期望 {args.expect}  {'OK' if n == args.expect else 'MISMATCH'}")
    print(f"[2] 空 TXT 数量     : {len(empty)}（无检测）")
    print(f"[3] 格式错误行      : {bad_format}")
    print(f"[4] 类别越界行      : {bad_class}")
    print(f"[5] 坐标越界行      : {bad_coord}")
    print(f"[6] 总检测框数      : {n_lines}")
    print(f"[7] 框数分布        : mean={boxes.mean():.2f} median={np.median(boxes):.0f} "
          f"max={boxes.max()} min={boxes.min()}")
    print(f"[8] 框数>0 图片数   : {(boxes > 0).sum()} / {n}")
    print(f"[9] 达 100 框上限   : {truncated} 张（每图最多写 100 框，触发截断）")
    print(f"[10] 单图最大框数   : {max_boxes_here}")

    # 打包 zip（archive 内为 results/<name>.txt，与 predict.py 一致）
    zip_path = Path(args.zip)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for txt in sorted(results_dir.glob("*.txt")):
            zf.write(txt, arcname=f"results/{txt.name}")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    print("-" * 64)
    print(f"submission.zip     : {zip_path}")
    print(f"  zip 内文件数     : {len(names)}（应 = {args.expect}）")
    print(f"  zip 大小         : {zip_path.stat().st_size / 1024:.1f} KB")
    print("=" * 64)

    ok = (n == args.expect and bad_format == 0 and bad_class == 0 and bad_coord == 0
          and len(names) == args.expect)
    print("结论:", "ALL PASS" if ok else "HAS ISSUES - see above")


if __name__ == "__main__":
    main()
