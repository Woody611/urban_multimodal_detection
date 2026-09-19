#!/bin/bash
# §8 NMS/max_det 敏感性：同一 checkpoint、同一验证集，只改推理 NMS 参数（只读诊断，不生成提交）
cd "$(dirname "$0")/../.." || exit 1
W=runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt
CFG=configs/train_rgbird_ir_quicktest.yaml
SRC=data/processed/rgbid_split/images/val/visible
O=diagnostic/small_object_error_attribution
for spec in "0.5:300" "0.6:300" "0.8:300"; do
  IOU="${spec%%:*}"; MD="${spec##*:}"
  echo "=== iou=$IOU max_det=$MD ==="
  python scripts/predict_rect.py --weights "$W" --train_config "$CFG" --source "$SRC" \
    --mode rect --imgsz 1280 --conf 0.001 --iou "$IOU" --max_det "$MD" --max_boxes 300 \
    --batch 16 --output "$O/nms_iou${IOU}_md${MD}" > "$O/_nms_iou${IOU}.log" 2>&1
  echo "  exit=$? txt=$(ls $O/nms_iou${IOU}_md${MD}/results 2>/dev/null | wc -l)"
done
echo "=== NMS variants DONE ==="
