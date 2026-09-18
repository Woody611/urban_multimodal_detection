#!/bin/bash
cd /d/gyt/AIC/urban_multimodal_detection
export PYTHONIOENCODING=utf-8
for i in $(seq 1 12); do
  n=$(ls diagnostic/rgbid_conf_0001_candidate/results/ | wc -l)
  if [ "$n" -ge 1000 ]; then echo "COMPLETE at $n"; break; fi
  python - <<'PY'
from pathlib import Path
src = Path('data/raw/test/visible')
done = {p.stem for p in Path('diagnostic/rgbid_conf_0001_candidate/results').glob('*.txt')}
todo = [p for p in sorted(src.iterdir())
        if p.suffix.lower() in {'.jpg','.jpeg','.png'} and p.stem not in done][:25]
Path('diagnostic/rgbid_conf_0001_candidate/_todo.txt').write_text(
    "\n".join(str(p.resolve()) for p in todo), encoding='utf-8')
PY
  timeout 600 python scripts/predict_rect.py \
    --weights runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt \
    --train_config configs/train_rgbird_ir_quicktest.yaml \
    --source diagnostic/rgbid_conf_0001_candidate/_todo.txt \
    --mode rect --imgsz 1280 --conf 0.0001 --iou 0.7 --max_det 300 --max_boxes 100 \
    --batch 4 --output diagnostic/rgbid_conf_0001_candidate >> diagnostic/rgbid_conf_0001_candidate/chunk_log.txt 2>&1
  echo "chunk $i done, total=$(ls diagnostic/rgbid_conf_0001_candidate/results/ | wc -l)"
done
