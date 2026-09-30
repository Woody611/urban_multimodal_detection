#!/bin/bash
# ============================================================================
# OASA scale=1.4 正式单变量实验 —— 云端执行脚本（必须在 GPU 机器上运行）
#
# 用法:  bash diagnostic/oasa_scale14/_cloud_run_oasa_s14.sh [项目根目录]
#
# 做的六件事（预注册式：一次跑完，中途不因曲线形状停训）：
#   0) 开训前 provenance：代码/配置 SHA256 校验、目标目录必须不存在
#   1) 补测 OASA scale=2.0 的**本地官方口径**（该数字此前从未算过，缺它无法做 §10 对比）
#   2) 训练 OASA scale=1.4，完整 300 epoch
#   3) 运行期合同核验（Mosaic ON / close_mosaic OFF / 算子统计闭合）
#   4) 独立 inference（冻结管线，与 baseline / 2.0 完全相同的口径）
#   5) 官方口径评测 + checkpoint provenance（best/last SHA256、best epoch）
#
# 本脚本不修改任何源码 / 配置 / 既有 checkpoint，不上传线上。
# ============================================================================
set -e

find_root() {
  local d; d="$(cd "$(dirname "$0")" && pwd)"
  while [ "$d" != "/" ] && [ -n "$d" ]; do
    if [ -d "$d/ultralytics" ] && [ -d "$d/configs" ]; then echo "$d"; return 0; fi
    d="$(dirname "$d")"
  done
  return 1
}
if [ -n "$1" ]; then ROOT="$1"; else ROOT="$(find_root)" || { echo "!! 无法定位项目根"; exit 1; }; fi
cd "$ROOT"
echo "项目根: $ROOT    $(date)"

# ---------------- 冻结常量（不得随实验改动）----------------
EXPECT_AUG=5cb9a407625921ce3f12ffc13df2d703f0a49d9ce1ba067edaf025bced96c518
EXPECT_CFG14=dee6601dac39309d091164ddf3ee0056
MODEL_CFG=configs/yolo11m_sepstem.yaml
CFG_14=configs/oasa_pre_mosaic_scale14.yaml
CFG_20=configs/oasa_pre_mosaic.yaml
RUN_14=runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_oasa_s14
RUN_20=runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_oasa
RUN_BASE=runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe
SPLIT=data/processed/rgbid_split_train
SRC=$SPLIT/images/val/visible
OUT=diagnostic/oasa_scale14
mkdir -p "$OUT" logs

# 冻结的官方评测链路（与既有 baseline 完全一致；--max_boxes 300 是冻结值，不要改）
eval_one () {  # $1=weights  $2=cfg  $3=tag
  local W="$1" CFG="$2" TAG="$3" D="$OUT/$3"
  mkdir -p "$D"
  echo "  [infer] $TAG  $(date)"
  python scripts/predict_rect.py --weights "$W" --train_config "$CFG" --source "$SRC" \
    --mode rect --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 300 \
    --batch 16 --output "$D" > "$OUT/_$TAG.infer.log" 2>&1
  echo "         txt=$(ls $D/results 2>/dev/null | wc -l)"
  echo "  [official_map] $TAG"
  python scripts/official_map.py --results "$D/results" --split_root "$SPLIT" \
    > "$OUT/_$TAG.official_map.txt" 2>&1
  grep -iE "official|mAP50-95|^ *R@|micro" "$OUT/_$TAG.official_map.txt" | head -20 || true
}

echo; echo "=== 0) 开训前 provenance ==="
AUG=$(sha256sum ultralytics/data/augment.py | cut -d' ' -f1)
[ "$AUG" = "$EXPECT_AUG" ] || { echo "!! augment.py SHA256 不符：$AUG"; exit 1; }
echo "  augment.py  SHA256 OK  ${AUG:0:32}"
CFG=$(sha256sum "$CFG_14" | cut -d' ' -f1)
[ "${CFG:0:32}" = "$EXPECT_CFG14" ] || { echo "!! oasa_pre_mosaic_scale14.yaml SHA256 不符：${CFG:0:32}"; exit 1; }
echo "  1.4 配置    SHA256 OK  ${CFG:0:32}"
[ -d "$RUN_14" ] && { echo "!! $RUN_14 已存在 —— ultralytics 会静默追加 '2'，先处理"; exit 1; }
echo "  目标目录 $RUN_14 不存在 OK"
git log -1 --format="  HEAD %H %ad %s" --date=short || true

echo; echo "  --- hard gates 1-7（动态验证：真实建 dataset / 真实 close_mosaic 入口 / 真实 val transforms）---"
python diagnostic/oasa_scale14/_hard_gates.py || { echo "!! hard gates 1-7 FAIL -> STOP"; exit 1; }
echo; echo "  --- single-variable args diff ---"
python diagnostic/oasa_scale14/_args_check.py || { echo "!! single-variable assertion FAIL -> STOP"; exit 1; }

echo; echo "=== 1) OASA scale=2.0 官方口径 —— 已于 2026-09-23 完成，勿重跑 ==="
echo "  OASA-2.0 official mAP50-95 = 0.51122   SHA256 922a9043…37991"
echo "  D' (cap=100, D' 基线)      = 0.51528   (OFFICIAL_EVAL_FREEZE_AND_HISTORICAL_RETEST.md:122)"
echo "  Δ(2.0 − D') = −0.00406"
echo "  [skip] 按 §0 冻结，不重训 / 不重评 OASA-2.0"

echo; echo "=== 2) 训练 OASA scale=1.4（300 epoch，不中途停训）$(date) ==="
python scripts/train.py --model_config "$MODEL_CFG" --train_config "$CFG_14" 2>&1 \
  | tee "logs/oasa_s14_$(date +%m%d_%H%M).log"
echo "  训练完成 $(date)"

echo; echo "=== 3) 运行期合同核验 ==="
AY=$RUN_14/args.yaml
echo "  -- args.yaml 关键项 --"
grep -E "^(epochs|ir_encoding|channels|batch|imgsz|close_mosaic|mosaic|name):" "$AY" || true
grep -E "^object_scale_aug" "$AY" || true
echo "  -- 算子运行期 marker（取训练日志）--"
L=$(ls -t logs/oasa_s14_*.log | head -1)
echo "  日志: $L"
grep -c "insertion=pre_mosaic" "$L" || true
grep -m2 "\[OASA\] enabled" "$L" || true
echo "  -- 末 10 epoch 必须 applied=0 / skip_nomosaic 满 --"
grep "\[OASA\]" "$L" | tail -8 || true
echo "  ⚠ 若上面末段出现 applied>0，说明 close_mosaic 期 OASA 未关闭 → 本轮结果无效，停止解读"

echo; echo "=== 4) 独立 inference + 官方评测（与 OASA-2.0 完全同一条命令）==="
echo "  --- evaluator regression tests（必须 ALL PASS）---"
python scripts/test_official_eval.py > "$OUT/_oasa14_best.test_official_eval.txt" 2>&1 \
  && echo "  [PASS] regression tests ALL PASS" \
  || { echo "!! evaluator regression FAIL -> STOP"; exit 1; }
echo "  --- 记录 best.pt 推理前 SHA256（推理后需相同）---"
sha256sum "$RUN_14/weights/best.pt" | tee "$OUT/_oasa14_best.ckpt_sha_before.txt"
eval_one "$RUN_14/weights/best.pt" "$CFG_14" oasa14_best
echo "  --- 确认推理未改动 checkpoint ---"
sha256sum "$RUN_14/weights/best.pt" | tee "$OUT/_oasa14_best.ckpt_sha_after.txt"
cmp -s "$OUT/_oasa14_best.ckpt_sha_before.txt" "$OUT/_oasa14_best.ckpt_sha_after.txt" \
  && echo "  [PASS] best.pt 未被改动" || { echo "!! checkpoint 被改动"; exit 1; }

echo; echo "=== 5) checkpoint provenance ==="
for w in best last; do
  f="$RUN_14/weights/$w.pt"
  [ -f "$f" ] && echo "  $w.pt  $(sha256sum "$f" | cut -d' ' -f1)  $(stat -c%s "$f") bytes"
done
python - "$RUN_14/results.csv" <<'PY'
import sys, csv
rows=list(csv.DictReader(open(sys.argv[1])))
col=[c for c in rows[0] if "mAP50-95" in c][0]
best=max(rows,key=lambda r: float(r[col]))
print(f"  epochs={len(rows)}  best {col}={float(best[col]):.5f} @ep{best['epoch']}")
PY

echo; echo "=== 完成 $(date) ==="
echo "回传本地做报告："
echo "  $OUT/_oasa20_best.official_map.txt  $OUT/_oasa14_best.official_map.txt"
echo "  $RUN_14/args.yaml  $RUN_14/results.csv  $L"
