#!/bin/bash
# 逐个等待消融推理完成，完成即跑官方扩展评测
cd "$(dirname "$0")/../.." || exit 1
D=diagnostic/modality_dropout

wait_and_eval () {  # $1=子目录名  $2=tag
  echo "[watch] 等待 $1 ..."
  while [ "$(ls $D/$1/results 2>/dev/null | wc -l)" -lt 400 ]; do sleep 30; done
  sleep 5
  echo "[watch] >>> 评测 $1 ($(date))"
  python scripts/modality_dropout_official_eval.py \
    --results "$D/$1/results" --split_root data/processed/rgbid_split \
    --tag "$2" --json "$D/eval_$1.json" 2>&1
  echo "[watch] <<< $1 完成 ($(date))"
}

wait_and_eval RGBID_md_ir_dropped    "New-IR-dropped"
wait_and_eval RGBID_md_depth_dropped "New-Depth-dropped"
wait_and_eval RGBID_base_ir_dropped  "Base-IR-dropped"
wait_and_eval RGBID_base_depth_dropped "Base-Depth-dropped"
echo "[watch] ALL EVALS DONE"
