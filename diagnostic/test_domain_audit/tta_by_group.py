"""按「来源族 x 分辨率」分组的 TTA 只读重评。

动机：族 N 是唯一含 640x360 的族，且分辨率在推理时**可直接观测**（无需 GT、无需分类器）。
若某 TTA 分支只在某个可观测子群上为正，就构成「可门控的部分 TTA」的数据依据。

严格只读：不推理、不加载权重、不训练、不改 evaluator（复用 official_map 原函数）。
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
RUNS = ["RGBID_baseline", "RGBID_tta_hflip", "RGBID_tta_multiscale", "RGBID_f4_ens_3to1"]


def family(stem):
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


def main():
    imgs = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in OM.IMG_EXTS)
    # 分辨率由文件尺寸决定；只读 header，不解码像素
    from PIL import Image
    group_of, res_of = {}, {}
    for i, p in enumerate(imgs):
        w, h = Image.open(p).size
        res_of[i] = f"{w}x{h}"
        f = family(p.stem)
        group_of[i] = f"{f}" if f != "N" else f"N_{'HR' if w >= 1920 else 'LR'}"

    groups = defaultdict(set)
    for i, g in group_of.items():
        groups[g].add(i)
    order = sorted(groups, key=lambda g: -len(groups[g]))
    print("groups:", {g: len(groups[g]) for g in order})
    print()

    out = {}
    for run in RUNS:
        rdir = GAIN / run / "results"
        if not rdir.is_dir():
            continue
        gt, preds, st = OM.collect(IMG_DIR, LBL_DIR, rdir, 0.0, True, OM.MAX_BOXES_PER_IMAGE)
        row = {"overall": {}, "groups": {}}
        for tag, (mk, tail, mt) in {"official": ("mean", "zero", "conf"), "fork": ("trapz", "ramp", "iou")}.items():
            e = OM.evaluate(gt, preds, mk, tail, mt, "A")
            row["overall"][f"{tag}_A"] = round(e["mAP50-95"], 5)
            eB = OM.evaluate(gt, preds, "mean", "zero", "conf", "B")
            row["overall"]["official_B"] = round(eB["mAP50-95"], 5)
        for g in order:
            ids = groups[g]
            gg = {c: {i: v for i, v in d.items() if i in ids} for c, d in gt.items()}
            pp = {c: [t for t in lst if t[0] in ids] for c, lst in preds.items()}
            row["groups"][g] = {
                "n_img": len(ids),
                "officialA": round(OM.evaluate(gg, pp, "mean", "zero", "conf", "A")["mAP50-95"], 5),
                "officialB": round(OM.evaluate(gg, pp, "mean", "zero", "conf", "B")["mAP50-95"], 5),
                "fork": round(OM.evaluate(gg, pp, "trapz", "ramp", "iou", "A")["mAP50-95"], 5),
            }
        out[run] = row

    base = out["RGBID_baseline"]
    for metric in ["officialA", "officialB", "fork"]:
        print(f"=== Δ vs RGBID_baseline ({metric}) ===")
        print(f'{"run":24s} {"overall":>9s} ' + " ".join(f"{g:>11s}" for g in order))
        print(f'{"  (n_img)":24s} {"":>9s} ' + " ".join(f"{len(groups[g]):>11d}" for g in order))
        for run, row in out.items():
            if run == "RGBID_baseline":
                continue
            ov = row["overall"][{"officialA": "official_A", "officialB": "official_B", "fork": "fork_A"}[metric]]
            bv = base["overall"][{"officialA": "official_A", "officialB": "official_B", "fork": "fork_A"}[metric]]
            ds = " ".join(f'{row["groups"][g][metric] - base["groups"][g][metric]:+11.5f}' for g in order)
            print(f'{run:24s} {ov-bv:+9.5f} {ds}')
        print()
        print(f'{"  absolute levels":24s}')
        for run, row in out.items():
            ds = " ".join(f'{row["groups"][g][metric]:11.5f}' for g in order)
            print(f'{run:24s} {row["overall"][{"officialA":"official_A","officialB":"official_B","fork":"fork_A"}[metric]]:9.5f} {ds}')
        print()

    (Path(__file__).resolve().parent / "tta_by_group.json").write_text(
        json.dumps({"groups": {g: len(groups[g]) for g in order}, "runs": out}, ensure_ascii=False, indent=1),
        encoding="utf-8")


if __name__ == "__main__":
    main()
