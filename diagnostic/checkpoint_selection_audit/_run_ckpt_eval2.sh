#!/bin/bash
# 只读 checkpoint 验证（缩减版：仅 290 / 240 / 260；280 已完成）
# 与正式 baseline 完全相同的官方链路：predict_rect.py --mode rect (冻结) -> official_map.py (冻结)
# 不训练、不修改任何 checkpoint / 配置 / 源码、不覆盖 best.pt、不上传线上。
cd "$(dirname "$0")/../.." || exit 1
W=runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights
CFG=configs/train_rgbird_ir_quicktest.yaml
SRC=data/processed/rgbid_split/images/val/visible
OUT=diagnostic/checkpoint_selection_audit

for ep in 290 240 260; do
  ck="$W/epoch$ep.pt"
  if [ ! -f "$ck" ]; then echo "[skip] $ck 不存在"; continue; fi
  if [ -f "$OUT/eval_ep$ep.json" ]; then echo "[skip] ep$ep 已有评测结果"; continue; fi
  echo "==================== epoch$ep ===================="
  echo "[1/2] 推理 $(date)"
  python scripts/predict_rect.py \
    --weights "$ck" --train_config "$CFG" --source "$SRC" \
    --mode rect --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 300 \
    --batch 16 --output "$OUT/ep$ep" > "$OUT/_ep$ep.infer.log" 2>&1
  echo "      推理 exit=$? txt=$(ls $OUT/ep$ep/results 2>/dev/null | wc -l)"

  echo "[2/2] 官方口径评测 $(date)"
  python scripts/official_map.py --results "$OUT/ep$ep/results" \
    --split_root data/processed/rgbid_split > "$OUT/_ep$ep.official_map.txt" 2>&1
  python scripts/modality_dropout_official_eval.py --results "$OUT/ep$ep/results" \
    --split_root data/processed/rgbid_split --tag "epoch$ep" \
    --json "$OUT/eval_ep$ep.json" > "$OUT/_ep$ep.eval.txt" 2>&1
  grep -E "^  official|^  micro|^  R@" "$OUT/_ep$ep.eval.txt"
done
echo "==================== DONE $(date) ===================="
