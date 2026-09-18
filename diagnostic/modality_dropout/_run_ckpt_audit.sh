#!/bin/bash
# §4 checkpoint 审计：等前面的消融链跑完后，用同一官方口径评测 baseline last.pt
cd "$(dirname "$0")/../.." || exit 1
D=diagnostic/modality_dropout

echo "[ckpt] 等待消融链结束 ..."
while ! grep -q "ALL DONE" "$D/_chain.log" 2>/dev/null; do sleep 30; done
echo "[ckpt] 消融链已结束，开始 baseline last.pt 推理 ($(date))"

python scripts/predict_rect.py \
  --weights runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/last.pt \
  --train_config configs/train_rgbird_ir_quicktest.yaml \
  --source data/processed/rgbid_split/images/val/visible \
  --mode rect --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 300 \
  --batch 16 --output "$D/RGBID_base_last" \
  > "$D/_RGBID_base_last.log" 2>&1
echo "[ckpt] last.pt 推理 exit=$? txt=$(ls $D/RGBID_base_last/results 2>/dev/null | wc -l)"

python scripts/modality_dropout_official_eval.py \
  --results "$D/RGBID_base_last/results" --split_root data/processed/rgbid_split \
  --tag "Base-LAST-ep300" --json "$D/eval_RGBID_base_last.json" 2>&1

echo "[ckpt] ALL DONE ($(date))"
