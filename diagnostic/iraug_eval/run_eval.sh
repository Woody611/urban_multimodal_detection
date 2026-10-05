set -u
SRC=data/processed/rgbid_split_train/images/val/visible
IMG=data/processed/rgbid_split_train/images/val/visible
LBL=data/processed/rgbid_split_train/labels/val/visible

run () {  # $1=tag  $2=weights  $3=train_config
  echo "================ $1  ($(date +%H:%M:%S)) ================"
  python scripts/predict_rect.py --weights "$2" --train_config "$3" \
      --source "$SRC" --mode rect --conf 0.001 --iou 0.7 --imgsz 1280 \
      --max_boxes 100 --max_det 300 --device cpu \
      --output "diagnostic/iraug_eval/$1" || { echo "PREDICT_FAIL $1"; return 1; }
  python -X utf8 scripts/official_eval.py --results "diagnostic/iraug_eval/$1/results" \
      --images "$IMG" --labels "$LBL" --metric A \
      --json "diagnostic/iraug_eval/$1/_official.json" || { echo "EVAL_FAIL $1"; return 1; }
  echo "---- $1 DONE ($(date +%H:%M:%S)) ----"
}

run Dp_best   runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt       configs/train_rgbid_sepstem_clahe.yaml
run IRAug_best runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_iraug/weights/best.pt configs/train_rgbid_sepstem_clahe_iraug.yaml
echo "ALL_DONE"
