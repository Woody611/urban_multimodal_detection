"""_mtime_scan.py — 全仓库「本轮开始后被写」的既有文件扫描（只读）"""
from __future__ import annotations

import datetime
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
CUT = (BASE / "_git_before.txt").stat().st_mtime   # 本轮开始时刻
SKIP = ("diagnostic/train_side_cls_supervision", ".git", "__pycache__", "AppData")

hits = []
for root, dirs, files in os.walk("."):
    r = root.replace(os.sep, "/")
    if any(s in r for s in SKIP):
        dirs[:] = []
        continue
    for f in files:
        p = os.path.join(root, f)
        try:
            m = os.path.getmtime(p)
        except OSError:
            continue
        if m > CUT:
            hits.append((datetime.datetime.fromtimestamp(m).strftime("%H:%M:%S"), p))
hits.sort()
print(f"以本轮开始时刻 {datetime.datetime.fromtimestamp(CUT)} 为界，之后被写过的既有文件数 = {len(hits)}")
for t, p in hits:
    print("   ", t, p.replace(os.sep, "/"))
