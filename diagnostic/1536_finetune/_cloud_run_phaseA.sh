#!/bin/bash
# ============================================================================
# 1536 微调 Phase A —— 云端执行脚本（必须在 GPU 机器上运行）
#
# 用法:  bash diagnostic/1536_finetune/_cloud_run_phaseA.sh <BATCH> [项目根目录]
#        BATCH 必须来自 feasibility_1536.py 的实测结果（无默认值 → 强制先探测）
#
# 做的五件事（§十八 顺序）：
#   0) 基线完整性校验（正式 best.pt SHA256 必须未变，否则立即中止）
#   1) 两个探针目录必须不存在（严禁覆盖正式目录）
#   2) 把实测 BATCH 写入两个新配置的 batch_size（只改这两个新建文件的一行）
#   3) Probe 1: lr0=5e-3 / Probe 2: lr0=5e-4，各 imgsz=1536、25 epoch
#   4) 每个探针跑 1536 官方评估（predict_rect --imgsz 1536 + official_map + 扩展评测）
#   5) 收尾再校验正式 best.pt 未被改动
#
# 绝不触碰 runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/
# 不修改任何既有源码；train.py 只接受 --model_config/--train_config，故 batch 由配置写入。
# ============================================================================
set -e
if [ -z "$1" ]; then
  echo "!! 必须显式传入 BATCH（先运行 feasibility_1536.py 实测）。例: bash $0 5"
  exit 1
fi
BATCH="$1"

find_root() {
  local d; d="$(cd "$(dirname "$0")" && pwd)"
  while [ "$d" != "/" ] && [ -n "$d" ]; do
    if [ -d "$d/ultralytics" ] && [ -d "$d/configs" ]; then echo "$d"; return 0; fi
    d="$(dirname "$d")"
  done
  return 1
}
if [ -n "$2" ]; then ROOT="$2"; else ROOT="$(find_root)" || { echo "!! 无法定位项目根"; exit 1; }; fi
cd "$ROOT"
echo "项目根: $ROOT   训练 batch = $BATCH"

BASE_W=runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights
BASE_SHA=f9dddbfaca4cb08bde31d082eb0a54d499b8441eb1f97c6134a028caa6d83e55
C1=configs/train_rgbird_1536_finetune.yaml
C2=configs/train_rgbird_1536_finetune_lr5e4.yaml
R1=runs/urban_multimodal_det_yolo11_rgbird_1536_finetune
R2=runs/urban_multimodal_det_yolo11_rgbird_1536_finetune_lr5e4
OUT=diagnostic/1536_finetune
SRC=data/processed/rgbid_split/images/val/visible

echo; echo "=== 0) 基线完整性校验（§十七）==="
python - "$BASE_W/best.pt" "$BASE_SHA" <<'PY'
import hashlib, sys
from pathlib import Path
p, exp = Path(sys.argv[1]), sys.argv[2]
got = hashlib.sha256(p.read_bytes()).hexdigest()
print("  best.pt", got)
if got != exp:
    print("  !! 正式 best.pt 已变动 —— 立即中止"); sys.exit(1)
print("  OK 正式 best.pt 未被改动")
PY

echo; echo "=== 1) 目标目录必须不存在（严禁覆盖正式目录）==="
for d in "$R1" "$R2"; do
  [ -d "$d" ] && { echo "  !! $d 已存在 —— 中止"; exit 1; }
  echo "  OK 不存在: $d"
done
mkdir -p "$OUT"

echo; echo "=== 2) 写入实测 batch=$BATCH 到两个新配置（仅改 batch_size 一行）==="
python - "$BATCH" "$C1" "$C2" <<'PY'
import re, sys
from pathlib import Path
bs, *cfgs = sys.argv[1:]
for c in cfgs:
    p = Path(c); t = p.read_text(encoding="utf-8")
    new, n = re.subn(r"^batch_size:\s*\d+.*$", f"batch_size: {bs}", t, count=1, flags=re.M)
    assert n == 1, f"{c}: 未找到唯一的 batch_size 行"
    p.write_text(new, encoding="utf-8")
    print(f"  {c}: batch_size -> {bs}")
PY

run_eval () {   # $1 = run 名  $2 = train_config
  echo "[eval] 1536 官方口径推理 $(date)"
  python scripts/predict_rect.py \
    --weights "runs/$1/weights/best.pt" --train_config "$2" --source "$SRC" \
    --mode rect --imgsz 1536 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 300 \
    --batch 8 --output "$OUT/$1_full" > "$OUT/_$1.infer.log" 2>&1
  echo "[eval] txt=$(ls $OUT/$1_full/results 2>/dev/null | wc -l)"
  python scripts/official_map.py --results "$OUT/$1_full/results" \
    --split_root data/processed/rgbid_split > "$OUT/_$1.official_map.txt" 2>&1
  python scripts/modality_dropout_official_eval.py --results "$OUT/$1_full/results" \
    --split_root data/processed/rgbid_split --tag "$1" \
    --json "$OUT/eval_$1.json" > "$OUT/_$1.eval.txt" 2>&1
  grep -E "^  official|^  micro|^  R@" "$OUT/_$1.eval.txt" || true
}

echo; echo "=== 3) Probe 1: lr0=5e-3  $(date) ==="
python scripts/train.py --model_config configs/yolo11m_earlyfusion.yaml --train_config "$C1"
echo "=== 4) Probe 1 官方评估 @1536 ==="
run_eval urban_multimodal_det_yolo11_rgbird_1536_finetune "$C1"

echo; echo "=== 5) Probe 2: lr0=5e-4  $(date) ==="
python scripts/train.py --model_config configs/yolo11m_earlyfusion.yaml --train_config "$C2"
echo "=== 6) Probe 2 官方评估 @1536 ==="
run_eval urban_multimodal_det_yolo11_rgbird_1536_finetune_lr5e4 "$C2"

echo; echo "=== 7) 收尾校验：正式 best.pt 与 submission 未被改动 ==="
python - "$BASE_W/best.pt" "$BASE_SHA" <<'PY'
import hashlib, sys
from pathlib import Path
p, exp = Path(sys.argv[1]), sys.argv[2]
got = hashlib.sha256(p.read_bytes()).hexdigest()
print("  best.pt", got, "OK" if got == exp else "!! 已变动")
sys.exit(0 if got == exp else 1)
PY
echo; echo "PHASE A DONE $(date)  —— 把 $OUT/ 与两个 run 的 args.yaml/results.csv 回传本地做报告"
