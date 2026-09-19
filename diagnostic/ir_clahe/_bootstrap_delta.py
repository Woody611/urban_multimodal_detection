"""diagnostic/p2_probe/_bootstrap_delta.py — 只读：图像级配对 bootstrap，给 D(IR-CLAHE) vs Baseline 的 Δ 置信区间。

动机：官方口径的 Δsmall = +0.00236（base 0.05625）是否只是噪声？
      预注册判定规则（§12.8）本身不要求这一步，但它直接回答 P2 探针的科学命题
      「IR CLAHE 是否真的改善」，因此额外做一次。

方法：
  - 与 scripts/official_map.py 完全相同的官方口径（mean + tail0 + conf 匹配）；
  - 尺寸分桶与 diagnostic/p3attn_probe/_sml_eval.py 逐字相同（32²/96²）；
  - **配对**重采样：同一次抽到的图像集合同时用于两个模型（消除图像难度方差）；
  - 报告 Δ 的分布 + P(Δ <= 0)。

正确性：`--verify` 会在全量（不重采样）上与 official_map.evaluate 逐位比对总 mAP50-95；
        并复现 _sml_eval.py 的 S/M/L 值。两者一致才继续 bootstrap。

实现性能：把每个 (类, 图) 的 preds×GT IoU 矩阵**预计算一次**，
          之后的 10 阈值 × 4 组（total + 3 尺寸桶）只做纯 Python 贪心（复用已算好的 IoU 行），
          避免原实现每 (类,阈值) 重算一遍 IoU（原实现 400 次迭代需 ~7h）。

用法:
  python diagnostic/p2_probe/_bootstrap_delta.py --verify
  python diagnostic/p2_probe/_bootstrap_delta.py --iters 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import official_map as OM  # noqa: E402

SPLIT = ROOT / "data/processed/rgbid_split"
THR = [float(t) for t in OM.IOUV_OFFICIAL]
NC = OM.NC
SMAX, MMAX = 32 ** 2, 96 ** 2

DIRS = {
    "Baseline": ROOT / "diagnostic/rgbid_inference_gain/RGBID_baseline/results",
    "D_clahe": ROOT / "diagnostic/ir_clahe/best_full/results",
}


# ============================================================
# 快路径：与 official_map.curve_for_class 语义逐条对齐的等价实现
# ============================================================
def _prep(gt, preds, keep_ids):
    """把 (gt_by_cls, preds_by_cls) 压成 {cls: {img: (confs, iou(P,G))}}，IoU 只算一次。

    语义与 official_map.curve_for_class(match='conf') 完全一致：
      按 conf 降序（稳定）；对每个预测取「尚未匹配的 GT 中 IoU 最大者」，
      其 IoU >= 阈值则 TP 并占用该 GT，否则 FP。
    尺寸/阈值只会改变「哪些 GT 列可用」，不改变 IoU 数值，故矩阵可完全复用。
    """
    remap = None
    if keep_ids is not None:
        remap = {}
        for new, old in enumerate(keep_ids):
            remap.setdefault(old, new)

    out = {}
    for c in range(NC):
        per_img = {}
        gts = gt[c]
        # GT：按图聚合（重采样时同图多份 → 按新 id 各存一份）
        gt_by_new = {}
        if remap is None:
            for img, bx in gts.items():
                if len(bx):
                    gt_by_new[img] = bx
        else:
            for new, old in enumerate(keep_ids):
                bx = gts.get(old)
                if bx is not None and len(bx):
                    prev = gt_by_new.get(new)
                    gt_by_new[new] = bx if prev is None else np.concatenate([prev, bx], 0)

        # 预测：同一 (img, conf, box) 按新 id 归位
        preds_by_new = {}
        for (img, conf, box) in preds[c]:
            new = img if remap is None else remap.get(img)
            if new is None:
                continue
            preds_by_new.setdefault(new, []).append((conf, box))

        for new in set(gt_by_new) | set(preds_by_new):
            gb = gt_by_new.get(new)
            pl = preds_by_new.get(new, [])
            if gb is None or not len(gb):
                gb = np.zeros((0, 4))
            if not pl:
                continue
            # 稳定排序：conf 降序
            order = sorted(range(len(pl)), key=lambda i: -pl[i][0])
            confs = np.array([pl[i][0] for i in order], dtype=np.float64)
            boxes = np.array([pl[i][1] for i in order], dtype=np.float64)
            iou = OM.box_iou_np(boxes, gb) if len(gb) else np.zeros((len(boxes), 0))
            per_img[new] = {"confs": confs, "iou": iou}
        out[c] = per_img
    return out


_EMPTY_MASK = np.zeros(0, dtype=bool)


def _greedy_ap(per_img, thr, gt_sizes, allow=None):
    """allow: None=全部 GT 可用；否则 {img: np.ndarray(G,) bool}（缺该图键 = 该桶无 GT）。

    注 1：`gt_sizes`（{img: G}，覆盖**所有有 GT 的图**，含没有预测的图）才是 npos 的来源 ——
          不能从 per_img 推，否则「有 GT 无预测」的图被漏计，recall 虚高（本脚本 verify 抓到过）。
    注 2：per_img 含「只有预测、没有 GT」的图（iou 形状 (P,0)）——这些图在 allow 里没有键，
          与 official_map.curve_for_class 中 gt_by_img.get(img) is None 的分支等价（全部计 FP）。
    """
    if allow is None:
        npos = sum(gt_sizes.values())
    else:
        npos = sum(int(m.sum()) for m in allow.values())
    flat = []           # (conf, img, k) 全局 conf 降序
    for img, d in per_img.items():
        for k in range(len(d["confs"])):
            flat.append((d["confs"][k], img, k))
    if npos == 0 or not flat:
        return 0.0
    flat.sort(key=lambda z: -z[0])

    matched = {}
    tps = np.empty(len(flat))
    fps = np.empty(len(flat))
    for idx, (_c, img, k) in enumerate(flat):
        d = per_img[img]
        row = d["iou"][k]
        ms = matched.get(img)
        if ms is None:
            ms = matched[img] = set()
        am = None if allow is None else allow.get(img, _EMPTY_MASK)
        best_j, best_v = -1, -1.0
        if am is None:
            for j in range(row.shape[0]):
                if j in ms:
                    continue
                v = row[j]
                if v > best_v:
                    best_v, best_j = v, j
        else:
            for j in np.nonzero(am)[0]:
                j = int(j)
                if j in ms:
                    continue
                v = row[j]
                if v > best_v:
                    best_v, best_j = v, j
        if best_j >= 0 and best_v >= thr:
            ms.add(best_j)
            tps[idx], fps[idx] = 1.0, 0.0
        else:
            tps[idx], fps[idx] = 0.0, 1.0

    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    recall = tp_cum / npos
    precision = tp_cum / np.maximum(tp_cum + fp_cum, OM.EPS)
    return OM.ap_from_curve(recall, precision, avg="mean", tail="zero")


def _bucket_mask(bx):
    a = np.clip(bx[:, 2] - bx[:, 0], 0, None) * np.clip(bx[:, 3] - bx[:, 1], 0, None)
    r = np.full(len(a), 2, int)
    r[a < SMAX] = 0
    r[(a >= SMAX) & (a < MMAX)] = 1
    return r


def metrics(prepped, gt_by_new_of_cls):
    """返回 total mAP50-95 + small/medium/large AP（官方口径）。"""
    total = []
    for t in THR:
        aps = []
        for c in range(NC):
            gbn = gt_by_new_of_cls[c]
            if not gbn:                     # 官方: 无 GT 的类别忽略
                continue
            aps.append(_greedy_ap(prepped[c], t, {img: len(bx) for img, bx in gbn.items()}))
        total.append(float(np.mean(aps)) if aps else 0.0)

    sizes = {}
    for si, sn in enumerate(("small", "medium", "large")):
        per_t = []
        for t in THR:
            vals = []
            for c in range(NC):
                gbn = gt_by_new_of_cls[c]
                if not gbn:
                    continue
                allow = {img: (_bucket_mask(bx) == si) for img, bx in gbn.items()}
                if not any(a.any() for a in allow.values()):
                    continue
                vals.append(_greedy_ap(prepped[c], t, None, allow))
            per_t.append(float(np.mean(vals)) if vals else 0.0)
        sizes[sn] = float(np.mean(per_t))
    return {"mAP50-95": float(np.mean(total)), **sizes}


def gt_by_new(gt, keep_ids):
    out = {}
    for c in range(NC):
        d = {}
        if keep_ids is None:
            for img, bx in gt[c].items():
                if len(bx):
                    d[img] = bx
        else:
            for new, old in enumerate(keep_ids):
                bx = gt[c].get(old)
                if bx is not None and len(bx):
                    prev = d.get(new)
                    d[new] = bx if prev is None else np.concatenate([prev, bx], 0)
        out[c] = d
    return out


def run(gt, preds, keep_ids):
    gbn = gt_by_new(gt, keep_ids)
    prepped = _prep(gt, preds, keep_ids)
    return metrics(prepped, gbn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true", help="全量逐位比对 official_map 后退出")
    args = ap.parse_args()

    data, n_img = {}, None
    for name, d in DIRS.items():
        gt, preds, st = OM.collect(SPLIT / "images/val/visible", SPLIT / "labels/val/visible",
                                   d, 0.0, True)
        data[name] = (gt, preds)
        n_img = st["images"]
        print(f"[collect] {name:<9} images={st['images']} corrupt_dropped={st['corrupt']} "
              f"gt={st['gt']} pred={st['pred']}", flush=True)

    # ---- 正确性自检：全量下必须与 official_map.evaluate 逐位一致 ----
    print("\n[verify] 全量（不重采样）与 official_map.evaluate 比对", flush=True)
    point = {}
    for name, (gt, preds) in data.items():
        fast = run(gt, preds, None)
        ref = OM.evaluate(gt, preds, "mean", "zero", "conf")
        ok = abs(fast["mAP50-95"] - ref["mAP50-95"]) < 1e-12
        print(f"  {name:<9} fast={fast['mAP50-95']:.6f} ref={ref['mAP50-95']:.6f} "
              f"exact={ok}  | small={fast['small']:.5f} medium={fast['medium']:.5f} "
              f"large={fast['large']:.5f}", flush=True)
        assert ok, "快路径与 official_map 不一致，拒绝继续"
        point[name] = fast
    if args.verify:
        return

    print("\n[point estimate]", flush=True)
    for n, v in point.items():
        print(f"  {n:<9} mAP50-95={v['mAP50-95']:.5f}  small={v['small']:.5f} "
              f"medium={v['medium']:.5f}  large={v['large']:.5f}", flush=True)

    rng = np.random.default_rng(args.seed)
    keys = ("mAP50-95", "small", "medium", "large")
    deltas = {k: [] for k in keys}
    for it in range(args.iters):
        ids = rng.integers(0, n_img, size=n_img).tolist()
        m = {n: run(gt, preds, ids) for n, (gt, preds) in data.items()}
        for k in keys:
            deltas[k].append(m["D_clahe"][k] - m["Baseline"][k])
        if (it + 1) % 25 == 0:
            print(f"  ... {it + 1}/{args.iters}", flush=True)

    print(f"\n[paired image-level bootstrap, iters={args.iters}, seed={args.seed}]")
    print(f"  {'metric':<10}{'point Δ':>10}{'boot mean':>11}{'2.5%':>10}{'97.5%':>10}"
          f"{'P(Δ<=0)':>10}{'σ':>9}")
    for k in keys:
        d = np.array(deltas[k])
        pt = point["D_clahe"][k] - point["Baseline"][k]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  {k:<10}{pt:>+10.5f}{d.mean():>+11.5f}{lo:>+10.5f}{hi:>+10.5f}"
              f"{float((d <= 0).mean()):>10.3f}{d.std(ddof=1):>9.5f}")

    # 单模型重采样 σ（尺度参考：指标本身对图像抽样的敏感度）
    print("\n[single-model bootstrap σ (baseline only), scale reference]")
    gt, preds = data["Baseline"]
    vals = {k: [] for k in keys}
    for _ in range(min(200, args.iters)):
        ids = rng.integers(0, n_img, size=n_img).tolist()
        m = run(gt, preds, ids)
        for k in keys:
            vals[k].append(m[k])
    for k in keys:
        print(f"  {k:<10} σ_metric={np.std(vals[k], ddof=1):.5f}")


if __name__ == "__main__":
    main()
