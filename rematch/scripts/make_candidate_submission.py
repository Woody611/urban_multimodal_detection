"""scripts/make_candidate_submission.py — 生成候选提交包（不自动上传）。

前置条件（全部满足才打包，否则拒绝并退出）：
  1. results 目录 TXT 数量 == --expect
  2. 合法性校验全部 PASS（scripts/validate_submission_txt.py 的同一套检查）

打包格式与冻结的 scripts/predict.py 完全一致：zip 内为 `results/<name>.txt`。

**本脚本不会上传任何内容，只生成本地 zip。**

用法:
  python scripts/make_candidate_submission.py \
      --results diagnostic/rgbid_inference_gain/TEST_ens_3to1/results \
      --source data/raw/test/visible --expect 1000 \
      --out submissions/rgbid_f4_ensemble_3to1_candidate
"""
from __future__ import annotations

import argparse
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ID_MIN, CLASS_ID_MAX = 0, 11


def validate(results_dir: Path, source_dir: Path, expect: int):
    """返回 (ok, report_dict)。检查项与 validate_submission_txt.py 一致。"""
    src_stems = [p.stem for p in source_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    txts = sorted(results_dir.glob("*.txt"))
    stems = [p.stem for p in txts]

    fatal = Counter()
    n_rows = 0
    conf_max = 0.0
    for txt in txts:
        lines = [ln for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            continue
        for ln in lines:
            p = ln.split()
            n_rows += 1
            if len(p) != 6:
                fatal["bad_format"] += 1
                continue
            try:
                cid_f, cx, cy, bw, bh, conf = (float(v) for v in p)
            except ValueError:
                fatal["bad_format"] += 1
                continue
            if any(math.isnan(v) or math.isinf(v) for v in (cid_f, cx, cy, bw, bh, conf)):
                fatal["nan_inf"] += 1
                continue
            conf_max = max(conf_max, conf)
            if conf > 1.0:
                fatal["conf_gt_1"] += 1
            if conf < 0.0:
                fatal["conf_negative"] += 1
            if int(cid_f) < CLASS_ID_MIN or int(cid_f) > CLASS_ID_MAX:
                fatal["class_out_of_range"] += 1
            if bw <= 0 or bh <= 0:
                fatal["wh_nonpositive"] += 1
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
                fatal["coord_out_of_range"] += 1
            if (cx - bw / 2 < -1e-6 or cx + bw / 2 > 1 + 1e-6
                    or cy - bh / 2 < -1e-6 or cy + bh / 2 > 1 + 1e-6):
                fatal["box_exceeds_image"] += 1

    missing = set(src_stems) - set(stems)
    extra = set(stems) - set(src_stems)
    dup = [s for s, c in Counter(stems).items() if c > 1]

    ok = (len(txts) == expect and not missing and not extra and not dup and sum(fatal.values()) == 0)
    rep = {"n_txt": len(txts), "n_src": len(src_stems), "n_rows": n_rows,
           "conf_max": conf_max, "missing": len(missing), "extra": len(extra),
           "dup": len(dup), "fatal": dict(fatal)}
    return ok, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--expect", type=int, default=1000)
    ap.add_argument("--out", required=True, help="候选包目录")
    ap.add_argument("--accept-degenerate-wh", action="store_true",
                    help="显式接受 w<=0 / h<=0 的退化框（冻结 pipeline 既有行为，非本次引入）。"
                         "不加此开关则拒绝打包。")
    args = ap.parse_args()

    results_dir = (PROJECT_ROOT / args.results).resolve()
    source_dir = (PROJECT_ROOT / args.source).resolve()
    out_dir = (PROJECT_ROOT / args.out).resolve()

    ok, rep = validate(results_dir, source_dir, args.expect)
    print("=" * 64)
    print("候选提交包前置校验")
    print("=" * 64)
    print(f"results   : {results_dir}")
    print(f"TXT 数量  : {rep['n_txt']} / 源图 {rep['n_src']} / 期望 {args.expect}")
    print(f"检测行数  : {rep['n_rows']}")
    print(f"conf 最大值: {rep['conf_max']:.6f}")
    print(f"遗漏/多余/重名: {rep['missing']} / {rep['extra']} / {rep['dup']}")
    print(f"致命项    : {rep['fatal'] if rep['fatal'] else '无'}")
    print(f"校验判定  : {'PASS' if ok else 'FAIL'}")
    print("=" * 64)

    # 退化框（w<=0 / h<=0）单独处理：这是冻结 pipeline 的既有行为，不是本次引入。
    # 只有显式 --accept-degenerate-wh 才放行，且必须打印证据。
    fatal = dict(rep["fatal"])
    n_wh = fatal.pop("wh_nonpositive", 0)
    residual = sum(fatal.values())
    if n_wh:
        print(f"[make] [WARN] 检出 {n_wh} 行退化框 (w<=0 或 h<=0)。")
        print("[make]    来源追溯：冻结 pipeline (predict.py::_format_lines) 对 scale_boxes")
        print("[make]    裁剪到图像边界外的框不设最小边长，历史提交文件同样含此现象")
        print("[make]    （RGBID test 冻结产出 155 行 / F4 test 72 行，且线上已评分）。")
        if not args.accept_degenerate_wh:
            print("[make]    未提供 --accept-degenerate-wh —— 拒绝生成提交包。")
            sys.exit(1)
        print("[make]    已显式提供 --accept-degenerate-wh，记录该例外并继续。")

    if residual:
        print(f"[make] 校验未通过（非退化框类致命项: {residual}）—— 拒绝生成提交包。")
        sys.exit(1)
    if not ok and not n_wh:
        print("[make] 校验未通过 —— 拒绝生成提交包。")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for txt in sorted(results_dir.glob("*.txt")):
            zf.write(txt, arcname=f"results/{txt.name}")

    with zipfile.ZipFile(zip_path) as zf:
        n_in = len(zf.namelist())
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[make] 候选包已生成: {zip_path}")
    print(f"[make] zip 内文件数: {n_in}  大小: {size_mb:.2f} MB")
    print("[make] 未上传。是否需要线上提交由用户明确授权。")


if __name__ == "__main__":
    main()
