#!/bin/bash
# 顺序执行模态消融推理（CPU，逐个跑避免线程争抢）
# 先等 New-Full 跑完，再依次：New-IR / New-Depth / Base-IR / Base-Depth
cd "$(dirname "$0")/../.." || exit 1
ROOT="$(pwd)"
echo "ROOT=$ROOT"

MD=runs/urban_multimodal_det_yolo11_rgbid_modality_dropout/weights/best.pt
BSE=runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt
CFG_MD=configs/train_rgbid_modality_dropout.yaml
CFG_BASE=configs/train_rgbird_ir_quicktest.yaml
SRC=data/processed/rgbid_split/images/val/visible

# --- 1) 等 New-Full 完成（结果目录 400 个 TXT）---
echo "[chain] 等待 New-Full 完成 ..."
while [ "$(ls diagnostic/modality_dropout/RGBID_md_full/results 2>/dev/null | wc -l)" -lt 400 ]; do
  sleep 30
done
echo "[chain] New-Full 完成"

run () {  # $1=weights $2=cfg $3=drop $4=outdir
  echo "[chain] >>> $4 (drop=$3)"
  python scripts/modality_ablation_infer.py \
    --weights "$1" --train_config "$2" --source "$SRC" \
    --drop "$3" --imgsz 1280 --conf 0.001 --iou 0.7 --max_det 300 --max_boxes 300 \
    --batch 16 --output "diagnostic/modality_dropout/$4" \
    > "diagnostic/modality_dropout/_${4}.log" 2>&1
  echo "[chain] <<< $4 exit=$? txt=$(ls diagnostic/modality_dropout/$4/results 2>/dev/null | wc -l)"
}

run "$MD"  "$CFG_MD"   ir    RGBID_md_ir_dropped
run "$MD"  "$CFG_MD"   depth RGBID_md_depth_dropped
run "$BSE" "$CFG_BASE" ir    RGBID_base_ir_dropped
run "$BSE" "$CFG_BASE" depth RGBID_base_depth_dropped

echo "[chain] ALL DONE"
