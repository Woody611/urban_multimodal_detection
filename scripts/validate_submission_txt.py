"""scripts/validate_submission_txt.py — 候选提交 TXT 合法性全量校验（只读）。

按最终审计要求逐项检查：
  文件数量 / 每行格式 / 类别 id 范围 / 坐标范围 / bbox 宽高 > 0 / conf ∈ [0,1] /
  NaN / Inf / conf > 1 / 负坐标 / 越界坐标 / 空文件 / 重复图片名 / 遗漏图片。

不静默修复任何内容：只报告事实。若发现 conf>1 会明确标记为 FAIL。

用法:
  python scripts/validate_submission_txt.py --results <dir>/results \
      --source data/raw/test/visible --expect 1000
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ID_MIN, CLASS_ID_MAX = 0, 11
NC = 12
N_FIELDS = 6


def main():
    ap = argparse.ArgumentParser(description="候选提交 TXT 合法性全量校验")
    ap.add_argument("--results", required=True)
    ap.add_argument("--source", required=True, help="原图目录（用于比对图片名）")
    ap.add_argument("--expect", type=int, default=1000)
    args = ap.parse_args()

    results_dir = (PROJECT_ROOT / args.results).resolve()
    source_dir = (PROJECT_ROOT / args.source).resolve()

    src_stems = [p.stem for p in source_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    txt_files = sorted(results_dir.glob("*.txt"))
    txt_stems = [p.stem for p in txt_files]

    stats = Counter()
    bad_examples = {}
    conf_max = 0.0
    dup_names = [s for s, c in Counter(txt_stems).items() if c > 1]

    for txt in txt_files:
        stem = txt.stem
        raw = txt.read_text(encoding="utf-8")
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if not lines:
            stats["empty_files"] += 1
            continue
        for li, ln in enumerate(lines, 1):
            parts = ln.split()
            stats["rows"] += 1
            if len(parts) != N_FIELDS:
                stats["bad_format"] += 1
                bad_examples.setdefault("bad_format", f"{stem}:{li} {ln!r}")
                continue
            try:
                vals = [float(v) for v in parts]
            except ValueError:
                stats["bad_format"] += 1
                bad_examples.setdefault("bad_format", f"{stem}:{li} {ln!r}")
                continue

            # NaN / Inf
            if any(math.isnan(v) for v in vals):
                stats["nan"] += 1
                bad_examples.setdefault("nan", f"{stem}:{li} {ln!r}")
                continue
            if any(math.isinf(v) for v in vals):
                stats["inf"] += 1
                bad_examples.setdefault("inf", f"{stem}:{li} {ln!r}")
                continue

            cid_f, cx, cy, bw, bh, conf = vals
            # 类别 id：必须是整数且在 [0,11]
            if not float(cid_f).is_integer():
                stats["class_non_integer"] += 1
            cid = int(cid_f)
            if cid < CLASS_ID_MIN or cid > CLASS_ID_MAX:
                stats["class_out_of_range"] += 1
                bad_examples.setdefault("class_out_of_range", f"{stem}:{li} {ln!r}")

            # conf
            conf_max = max(conf_max, conf)
            if conf > 1.0:
                stats["conf_gt_1"] += 1
                bad_examples.setdefault("conf_gt_1", f"{stem}:{li} {ln!r}")
            if conf < 0.0:
                stats["conf_negative"] += 1
                bad_examples.setdefault("conf_negative", f"{stem}:{li} {ln!r}")

            # 宽高
            if bw <= 0 or bh <= 0:
                stats["wh_nonpositive"] += 1
                bad_examples.setdefault("wh_nonpositive", f"{stem}:{li} {ln!r}")

            # 归一化坐标范围
            if cx < 0 or cx > 1 or cy < 0 or cy > 1 or bw > 1 or bh > 1:
                stats["coord_out_of_range"] += 1
                bad_examples.setdefault("coord_out_of_range", f"{stem}:{li} {ln!r}")
            # 负坐标
            if cx < 0 or cy < 0 or bw < 0 or bh < 0:
                stats["negative_coord"] += 1
                bad_examples.setdefault("negative_coord", f"{stem}:{li} {ln!r}")
            # 框体是否真正落在图内（cx±w/2 ∈ [0,1]）
            if (cx - bw / 2 < -1e-6 or cx + bw / 2 > 1 + 1e-6
                    or cy - bh / 2 < -1e-6 or cy + bh / 2 > 1 + 1e-6):
                stats["box_exceeds_image"] += 1
                bad_examples.setdefault("box_exceeds_image", f"{stem}:{li} {ln!r}")

    missing = set(src_stems) - set(txt_stems)
    extra = set(txt_stems) - set(src_stems)

    print("=" * 72)
    print("候选提交 TXT 合法性全量校验")
    print("=" * 72)
    print(f"results 目录     : {results_dir}")
    print(f"源图目录         : {source_dir}")
    print("-" * 72)
    print(f"源图数量         : {len(src_stems)}")
    print(f"TXT 文件数量     : {len(txt_files)}   (期望 {args.expect})")
    print(f"重复图片名       : {len(dup_names)}")
    print(f"遗漏图片         : {len(missing)}")
    print(f"多余图片         : {len(extra)}")
    print(f"空文件数量       : {stats['empty_files']}")
    print(f"总检测框行数     : {stats['rows']}")
    print(f"conf 最大值      : {conf_max:.6f}")
    print("-" * 72)
    checks = [
        ("行格式非法 (len!=6)", stats["bad_format"]),
        ("类别 id 非整数", stats["class_non_integer"]),
        ("类别 id 越界 [0,11]", stats["class_out_of_range"]),
        ("confidence > 1", stats["conf_gt_1"]),
        ("confidence < 0", stats["conf_negative"]),
        ("宽或高 <= 0", stats["wh_nonpositive"]),
        ("坐标越界 [0,1]", stats["coord_out_of_range"]),
        ("负坐标", stats["negative_coord"]),
        ("框体超出图像边界", stats["box_exceeds_image"]),
        ("NaN 行", stats["nan"]),
        ("Inf 行", stats["inf"]),
    ]
    for name, v in checks:
        print(f"  {name:<26}: {v}{'  <-- FAIL' if v else '  OK'}")
    if bad_examples:
        print("-" * 72)
        print("异常样例（前若干条）:")
        for k, v in bad_examples.items():
            print(f"  [{k}] {v}")
    print("-" * 72)

    fatal = (stats["bad_format"] + stats["class_non_integer"] + stats["class_out_of_range"]
             + stats["conf_gt_1"] + stats["conf_negative"] + stats["wh_nonpositive"]
             + stats["coord_out_of_range"] + stats["negative_coord"] + stats["nan"] + stats["inf"])
    count_ok = (len(txt_files) == args.expect and not missing and not extra and not dup_names)
    print(f"数量/完整性      : {'PASS' if count_ok else 'FAIL'}")
    print(f"内容合法性       : {'PASS' if fatal == 0 else 'FAIL'}")
    print(f"总体判定         : {'PASS' if (count_ok and fatal == 0) else 'FAIL'}")
    print("=" * 72)
    return 0 if (count_ok and fatal == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
