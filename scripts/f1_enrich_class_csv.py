"""scripts/f1_enrich_class_csv.py — 给 reports/f1_class_metrics.csv 补上「主要误检/漏检类别」列。

数据来源：reports/f1_class_metrics.csv（val 逐类指标 + GT 统计）
         reports/f1_confusion_matrix.json（matrix[pred, gt]，最后一行/列 = background）
追加列：
  det_rate     conf=0.001/IoU=0.45 下的检测率 = 命中 / GT（硬漏检下限）
  miss_to      本类 GT 的主要去向（含 background）Top-3，如 "background(272);animal(3)"
  fp_from      本类预测主要覆盖了谁（含 background）Top-3
  cross_cls     本类参与的跨类误判总量（不含 background）
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
CSVP = REP / "f1_class_metrics.csv"
CMP = REP / "f1_confusion_matrix.json"

rows = list(csv.DictReader(CSVP.open(encoding="utf-8")))
names = [r["class"] for r in rows]
nc = len(rows)

cm = json.load(CMP.open(encoding="utf-8"))
M = cm["matrix"]
# matrix[pred][gt]
def g(p, g_):
    return M[p][g_]

def top3(pairs, k=3):
    pairs = [(n, v) for n, v in pairs if v > 0]
    pairs.sort(key=lambda x: -x[1])
    return ";".join(f"{n}({v})" for n, v in pairs[:k]) or "-"

for i, r in enumerate(rows):
    gt_total = sum(g(p, i) for p in range(nc + 1))          # 本类 GT 总数（列 i）
    hit = g(i, i)
    r["det_rate"] = f"{hit / gt_total:.3f}" if gt_total else "-"
    # 本类 GT 的去向（列 i 的所有 pred）
    r["miss_to"] = top3([("background" if p == nc else names[p], g(p, i))
                         for p in range(nc + 1) if p != i])
    # 本类预测覆盖了谁（行 i 的所有 gt）
    r["fp_from"] = top3([("background" if gg == nc else names[gg], g(i, gg))
                         for gg in range(nc + 1) if gg != i])
    r["cross_cls"] = sum(g(i, gg) for gg in range(nc) if gg != i) + \
                     sum(g(p, i) for p in range(nc) if p != i)

fields = list(rows[0].keys())
with CSVP.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"[csv] 已更新 {CSVP}")
print("  " + ",".join(fields))
for r in rows:
    print(f"  {r['class']:<12} det_rate={r['det_rate']:<6} cross={r['cross_cls']:<3} "
          f"miss_to={r['miss_to']:<28} fp_from={r['fp_from']}")
