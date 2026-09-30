"""对既有 TTA / ensemble 落盘预测做「按来源族分层」的只读重评。

严格只读：
  - 不重新推理、不加载权重、不训练、不写提交、不修改 evaluator。
  - **复用** scripts/official_map.py 的 collect / evaluate 原函数（未改动其一行），
    只对 img_id 做子集切分。

动机：split_val 的来源族构成与 test 差异巨大（shuming 20.2% vs 5.4%），
若各族对 TTA 的响应不同号，val 上测得的 TTA 增益就无法迁移到 test。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import official_map as OM  # noqa: E402

SPLIT = ROOT / "data/processed/rgbid_split"
IMG_DIR = SPLIT / "images/val/visible"
LBL_DIR = SPLIT / "labels/val/visible"
GAIN = ROOT / "diagnostic/rgbid_inference_gain"

RUNS = ["RGBID_baseline", "RGBID_tta_hflip", "RGBID_tta_multiscale",
        "F4_baseline", "RGBID_f4_ens_3to1"]
FAMILY_OF = {}
IMG_ORDER = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in OM.IMG_EXTS)


def family(stem: str) -> str:
    if stem.startswith("shuming_"):
        return "shuming_*"
    if stem.startswith("hehe_"):
        return "hehe_*"
    if "suppl" in stem:
        return "*suppl*"
    if stem.count("_") == 2:
        return "N_N_N"
    if stem.count("_") == 0:
        return "N"
    return "other"


def subset(gt, preds, ids):
    g = {c: {i: v for i, v in d.items() if i in ids} for c, d in gt.items()}
    p = {c: [t for t in lst if t[0] in ids] for c, lst in preds.items()}
    return g, p


def main():
    for i, p in enumerate(IMG_ORDER):
        FAMILY_OF[i] = family(p.stem)
    fam_ids = defaultdict(set)
    for i, f in FAMILY_OF.items():
        fam_ids[f].add(i)
    n_img = len(IMG_ORDER)
    print(f"val images: {n_img}")
    fam_sizes = {k: len(v) for k, v in sorted(fam_ids.items(), key=lambda x: -len(x[1]))}
    print("family sizes:", fam_sizes)

    # 用像素尺寸顺手记录每族的分辨率构成
    import cv2
    fam_res = defaultdict(lambda: defaultdict(int))
    for i, p in enumerate(IMG_ORDER):
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        fam_res[FAMILY_OF[i]][f"{w}x{h}"] += 1
    print("family x resolution:", {k: dict(v) for k, v in fam_res.items()})
    print()

    results = {}
    for run in RUNS:
        rdir = GAIN / run / "results"
        if not rdir.is_dir():
            print("skip (missing)", run)
            continue
        gt, preds, st = OM.collect(IMG_DIR, LBL_DIR, rdir, 0.0, True, OM.MAX_BOXES_PER_IMAGE)
        entry = {"stats": st, "overall": {}, "per_family": {}}
        for tag, (mk, tail) in {"official": ("mean", "zero"), "fork": ("trapz", "ramp")}.items():
            m_gt, m_pr = gt, preds
            e = OM.evaluate(m_gt, m_pr, mk, tail, "conf" if tag == "official" else "iou", "A")
            entry["overall"][f"{tag}_mAP50_95"] = round(e["mAP50-95"], 5)
            entry["overall"][f"{tag}_mAP50"] = round(e["mAP50"], 5)
            entry["overall"][f"{tag}_mAP75"] = round(e["mAP75"], 5)
        # A 口径 per-family（固定 12 类分母）+ B 口径（仅有 GT 的类）
        for fam, ids in fam_ids.items():
            g, p = subset(gt, preds, ids)
            row = {"n_img": len(ids)}
            eA = OM.evaluate(g, p, "mean", "zero", "conf", "A")
            eB = OM.evaluate(g, p, "mean", "zero", "conf", "B")
            fA = OM.evaluate(g, p, "trapz", "ramp", "iou", "A")
            row.update({"officialA_mAP50_95": round(eA["mAP50-95"], 5),
                        "officialA_mAP50": round(eA["mAP50"], 5),
                        "officialA_mAP75": round(eA["mAP75"], 5),
                        "officialB_mAP50_95": round(eB["mAP50-95"], 5),
                        "fork_mAP50_95": round(fA["mAP50-95"], 5)})
            entry["per_family"][fam] = row
        results[run] = entry
        o = entry["overall"]
        print(f'{run:26s} official {o["official_mAP50_95"]:.5f}  mAP50 {o["official_mAP50"]:.5f} '
              f' mAP75 {o["official_mAP75"]:.5f} | fork {o["fork_mAP50_95"]:.5f}')

    print("\n=== per-family official mAP50-95 (metric A) ===")
    fams = [f for f, _ in sorted(fam_ids.items(), key=lambda x: -len(x[1]))]
    hdr = f'{"run":26s} ' + " ".join(f"{f:>10s}" for f in fams)
    print(hdr)
    for run, e in results.items():
        print(f'{run:26s} ' + " ".join(f'{e["per_family"][f]["officialA_mAP50_95"]:10.5f}' for f in fams))

    print("\n=== Δ vs RGBID_baseline (official mAP50-95, metric A) ===")
    base = results.get("RGBID_baseline")
    if base:
        for run, e in results.items():
            if run == "RGBID_baseline":
                continue
            d_all = e["overall"]["official_mAP50_95"] - base["overall"]["official_mAP50_95"]
            ds = " ".join(
                f'{e["per_family"][f]["officialA_mAP50_95"] - base["per_family"][f]["officialA_mAP50_95"]:+10.5f}'
                for f in fams)
            print(f'{run:26s} overall {d_all:+8.5f} | {ds}')
        print("\n=== Δ vs RGBID_baseline (fork mAP50-95, metric A) ===")
        for run, e in results.items():
            if run == "RGBID_baseline":
                continue
            d_all = e["overall"]["fork_mAP50_95"] - base["overall"]["fork_mAP50_95"]
            ds = " ".join(
                f'{e["per_family"][f]["fork_mAP50_95"] - base["per_family"][f]["fork_mAP50_95"]:+10.5f}'
                for f in fams)
            print(f'{run:26s} overall {d_all:+8.5f} | {ds}')

    print("\n=== family share: val vs test ===")
    print("val :", {k: f"{v/n_img:.2%}" for k, v in fam_sizes.items()})
    print("test:", {"N": "73.0%", "N_N_N": "18.0%", "shuming_*": "5.4%", "*suppl*": "2.0%", "hehe_*": "1.6%"})
    # 用 val 的 per-family 指标按「val 权重」和「test 权重」分别加权，估计重加权效应
    tw = {"N": 0.730, "N_N_N": 0.180, "shuming_*": 0.054, "*suppl*": 0.020, "hehe_*": 0.016}
    print("\n=== 同一 val 预测、两种家族权重下的 official mAP50-95（重加权反事实）===")
    print(f'{"run":26s} {"val-weight":>11s} {"test-weight":>12s} {"Δ(reweight)":>13s}')
    for run, e in results.items():
        vw = sum(e["per_family"][f]["officialA_mAP50_95"] * len(fam_ids[f]) / n_img for f in fams)
        xw = sum(e["per_family"][f]["officialA_mAP50_95"] * tw.get(f, 0.0) for f in fams)
        print(f'{run:26s} {vw:11.5f} {xw:12.5f} {xw-vw:+13.5f}')

    (Path(__file__).resolve().parent / "tta_by_family.json").write_text(
        json.dumps({"family_sizes": fam_sizes, "family_resolution": {k: dict(v) for k, v in fam_res.items()},
                    "runs": results}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
