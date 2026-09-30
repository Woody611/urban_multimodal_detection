#!/bin/bash
# ============================================================================
# 本机（CPU）冻结官方评测器 —— 与 OASA-2.0 锚点 0.51122 逐字节同一条命令
#
# 用法: bash diagnostic/oasa_scale14/_local_official_eval.sh <weights.pt> <train_config> <tag>
#   例: bash diagnostic/oasa_scale14/_local_official_eval.sh \
#         runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_oasa_s14/weights/best.pt \
#         configs/oasa_pre_mosaic_scale14.yaml  oasa14_best
#
# 只读：不训练、不修改 checkpoint / evaluator / inference / max_boxes。
# 冻结项（不得改动）：mode=rect imgsz=1280 conf=0.001 iou=0.7 max_det=300 max_boxes=300 batch=16
#                      evaluator 内部 MAX_BOXES_PER_IMAGE=100 不得覆盖
# ============================================================================
set -e
cd "$(dirname "$0")/../.." || exit 1

W="$1"; CFG="$2"; TAG="$3"
[ -z "$TAG" ] && { echo "用法: $0 <weights.pt> <train_config> <tag>"; exit 1; }
SPLIT=data/processed/rgbid_split_train
SRC=$SPLIT/images/val/visible
OUT=diagnostic/oasa_scale14/$TAG
LOGD=diagnostic/oasa_scale14
mkdir -p "$OUT"

echo "================= $TAG ================="
echo "[ckpt] $W"
echo "[ckpt] sha256_before = $(sha256sum "$W" | cut -d' ' -f1)"

echo "[1/3] 独立 inference（LoadImagesAndVideos + LetterBox，OASA OFF）$(date)"
python scripts/predict_rect.py --weights "$W" --train_config "$CFG" --source "$SRC" \
  --mode rect --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 300 \
  --batch 16 --output "$OUT" > "$LOGD/_$TAG.infer.log" 2>&1
echo "      txt=$(ls "$OUT/results" 2>/dev/null | wc -l) / 400"
grep -a "use_simotm" "$LOGD/_$TAG.infer.log" || true

echo "[2/3] evaluator regression tests $(date)"
python scripts/test_official_eval.py > "$LOGD/_$TAG.test_official_eval.txt" 2>&1 \
  && echo "      ALL PASS" || { echo "      !! FAIL -> 停止解读"; exit 1; }

echo "[3/3] official_map.py（cap=100 冻结默认，不覆盖）$(date)"
python scripts/official_map.py --results "$OUT/results" --split_root "$SPLIT" \
  > "$LOGD/_$TAG.official_map.txt" 2>&1

echo "[ckpt] sha256_after  = $(sha256sum "$W" | cut -d' ' -f1)"
echo "================= 结果（原始行）================="
cat "$LOGD/_$TAG.official_map.txt"
