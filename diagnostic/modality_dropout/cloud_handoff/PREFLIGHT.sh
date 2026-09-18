#!/bin/bash
# 云端开训前自检 —— 用「内容标记」校验，不依赖绝对 SHA256
# （云端与本地是两次独立 checkout，绝对哈希天然不同，不可用作判据）
# 用法: bash PREFLIGHT.sh [项目根目录]
set -e
# --- 健壮定位项目根：从脚本所在目录逐级上溯，找到含 ultralytics/ 与 configs/ 的目录 ---
find_root() {
  local d
  d="$(cd "$(dirname "$0")" && pwd)"
  while [ "$d" != "/" ] && [ -n "$d" ]; do
    if [ -d "$d/ultralytics" ] && [ -d "$d/configs" ]; then echo "$d"; return 0; fi
    d="$(dirname "$d")"
  done
  return 1
}

if [ -n "$1" ]; then ROOT="$1"; else ROOT="$(find_root)" || { echo "!! 无法自动定位项目根，请显式传入: bash $0 /path/to/project"; exit 1; }; fi
cd "$ROOT"
echo "项目根目录: $ROOT"
echo

echo "=== 1) 内容标记校验（真正决定实验是否有效的检查）==="
python - <<'PY'
from pathlib import Path
import yaml, sys

fails = []
def need(path, markers, must_have=True, why=""):
    p = Path(path)
    if not p.exists():
        print(f"  [FAIL] {path} 不存在   {why}"); fails.append(path); return
    t = p.read_text(encoding='utf-8', errors='replace')
    for m in markers:
        ok = (m in t)
        if ok != must_have:
            print(f"  [FAIL] {path}  {'缺少' if must_have else '意外含有'}: {m!r}   {why}")
            fails.append(f"{path}:{m}")
        else:
            print(f"  [ OK ] {path}  {'含有' if must_have else '不含'}: {m}")

need("ultralytics/data/base.py",
     ["apply_modality_dropout", "self.hyp = hyp", "MODALITY_DROPOUT_STATS"],
     True, "★核心：dropout 实现 + config 读取")
need("ultralytics/cfg/default.yaml", ["modality_dropout"], True, "默认键")
need("scripts/train.py",           ["modality_dropout"], True, "参数转发")
# 防止我早期那个错误补丁残留在 dataset.py 里
need("ultralytics/data/dataset.py", ["apply_modality_dropout"], False, "不应含（早期误改已回滚）")

# 新配置文件取值校验
p = Path("configs/train_rgbid_modality_dropout.yaml")
if not p.exists():
    print(f"  [FAIL] {p} 不存在"); fails.append(str(p))
else:
    cfg = yaml.safe_load(open(p, encoding='utf-8'))
    md  = cfg.get("modality_dropout", {})
    checks = [("enabled", True), ("ir_drop_prob", 0.075), ("depth_drop_prob", 0.075),
              ("ir_fill", 125), ("depth_fill", 0), ("mutually_exclusive", True),
              ("apply_train_only", True)]
    for k, v in checks:
        got = md.get(k)
        ok = (got == v)
        print(f"  [{'OK  ' if ok else 'FAIL'}] config.modality_dropout.{k} = {got}  (期望 {v})")
        if not ok: fails.append(f"config.{k}")
    name_ok = cfg.get("experiment_name") == "urban_multimodal_det_yolo11_rgbid_modality_dropout"
    print(f"  [{'OK  ' if name_ok else 'FAIL'}] experiment_name = {cfg.get('experiment_name')}")
    if not name_ok: fails.append("experiment_name")

print()
if fails:
    print("  => 校验未通过。请在云端执行:  bash deploy.sh <项目根目录>")
    sys.exit(1)
print("  => 内容标记全部通过")
PY

echo
echo "=== 2) GPU 可用性 ==="
python -c "import torch;assert torch.cuda.is_available(),'无 GPU';print('  GPU:',torch.cuda.get_device_name(0))"

echo
echo "=== 3) 目标目录不应存在 ==="
test ! -d runs/urban_multimodal_det_yolo11_rgbid_modality_dropout \
  && echo "  OK 目标目录不存在" \
  || { echo "  !! 目标目录已存在，请改名以免覆盖"; exit 1; }

echo
echo "=== 4) 最终验证 A-E（逐像素确认填充值）==="
python scripts/modality_dropout_final_check.py

echo
echo "============================================================"
echo "PREFLIGHT 全部通过 -> 可启动训练"
echo "python scripts/train.py --model_config configs/yolo11m_earlyfusion.yaml --train_config configs/train_rgbid_modality_dropout.yaml"
echo "============================================================"
