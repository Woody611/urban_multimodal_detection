#!/bin/bash
# 把 handoff 里的 6 个文件部署到正确位置，并校验 SHA256。
# 用法: 把本目录整体放到云端项目的任意位置，然后  bash deploy.sh /path/to/project
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
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
echo "项目根目录: $ROOT"
echo "handoff 源: $SRC"
echo

declare -A MAP=(
  ["base.py"]="ultralytics/data/base.py"
  ["default.yaml"]="ultralytics/cfg/default.yaml"
  ["train.py"]="scripts/train.py"
  ["train_rgbid_modality_dropout.yaml"]="configs/train_rgbid_modality_dropout.yaml"
  ["modality_dropout_final_check.py"]="scripts/modality_dropout_final_check.py"
  ["modality_dropout_smoke_test.py"]="scripts/modality_dropout_smoke_test.py"
)
for src in "${!MAP[@]}"; do
  dst="${MAP[$src]}"
  mkdir -p "$(dirname "$dst")"
  cp "$SRC/$src" "$dst"
  echo "  deployed  $src  ->  $dst"
done
echo
echo "=== SHA256 校验（仅用于本机自检；跨机器请以 PREFLIGHT 的内容标记为准）==="
python - <<'PY'
import hashlib
from pathlib import Path
exp = {
 "ultralytics/data/base.py":                 "caab122e5d1da3bbc17b753fd02e2869d554594e03a0cb2d26f240ed51b62cd8",
 "ultralytics/cfg/default.yaml":             "348737529cded11bc3d6684393df89698d827f6ba16d5e512a9a622fb1d88959",
 "scripts/train.py":                         "5ab210ec034fd02be481ec935507741abbafc371816ad7c1adc3c8bf50fc0bc8",
 "configs/train_rgbid_modality_dropout.yaml":"cb29b4aacfedfc34f2ed852acd7a0c6d530d443e68cee1912c7a6d6dbf498c7b",
 "scripts/modality_dropout_final_check.py":  "8d34f2c0079e48548d6464859e65aaee607f6eeed43888a485ec8380d626bffa",
}
ok = True
for f in exp:
    p = Path(f)
    got = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "缺失"
    good = got == exp[f]; ok &= good
    print(f"  {'OK  ' if good else 'DIFF'} {f}   {got[:16]}")
print("  =>", "部署成功" if ok else "校验失败，请检查")
raise SystemExit(0 if ok else 1)
PY


# ---------------------------------------------------------------
# 备选：仅打补丁（不覆盖整个文件，保留云端独有改动）
#   cd <项目根目录>
#   git apply cloud_handoff/modality_dropout.patch    # 或 patch -p1 < ...
# 注意：patch 基于「本地改前版本」生成，若云端同区域已被改动会 apply 失败，
#       届时请改用本脚本的整文件覆盖方式（deploy.sh）。
# ---------------------------------------------------------------
