"""_v3_analysis.py — V3：对 V2 的 38 个 E1 目标做 density / competition / border / matrix / class 分析。

只读。复用 V2 的 records 与 official_eval 口径，不生成新 prediction、不改核心代码。
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "diagnostic/small_object_cause_v2"))

from _v2_records import EPS0, IMAGES, LABELS, NAMES, SMALL_A, Dp, iter_split  # noqa: E402
from official_eval import (  # noqa: E402
    MAX_BOXES_PER_IMAGE, apply_max_boxes, box_iou_np, norm_xywh_to_xyxy, read_pred_txt,
)

OUT = Path(__file__).resolve().parent
RADII = (1, 2, 3, 5)


def load():
    gt = {}
    for stem, w, h, b, c in iter_split(IMAGES, LABELS):
        gt[stem] = dict(w=w, h=h, b=b, c=c)
    pred = {}
    for stem, v in gt.items():
        _, p, _ = read_pred_txt(Dp / f"{stem}.txt")
        p = apply_max_boxes(p, MAX_PRED := MAX_BOXES_PER_IMAGE)
        if len(p):
            pb, pc = norm_xywh_to_xyxy(p, v["w"], v["h"])
            pred[stem] = dict(box=pb, cls=pc, conf=p[:, 5].astype(np.float32))
        else:
            pred[stem] = dict(box=np.zeros((0, 4), np.float32), cls=np.zeros(0, int),
                              conf=np.zeros(0, np.float32))
    return gt, pred


def annotate(gt, pred):
    """为每个 GT 算：detected / any_overlap / 全局密度 / 多半径局部密度 / 边界。"""
    A = {}
    for stem, v in gt.items():
        if v["b"] is None or len(v["b"]) == 0:
            continue
        P = pred[stem]
        n = len(v["b"])
        bw = np.clip(v["b"][:, 2] - v["b"][:, 0], 0, None)
        bh = np.clip(v["b"][:, 3] - v["b"][:, 1], 0, None)
        a = bw * bh
        dg = np.sqrt(bw ** 2 + bh ** 2)
        ctr = np.stack([(v["b"][:, 0] + v["b"][:, 2]) / 2, (v["b"][:, 1] + v["b"][:, 3]) / 2], 1)
        dist = np.sqrt(((ctr[:, None, :] - ctr[None, :, :]) ** 2).sum(-1))
        det = np.zeros(n, bool); bs = np.zeros(n, np.float32); ba = np.zeros(n, np.float32)
        for j in range(n):
            s = np.where(P["cls"] == v["c"][j])[0] if len(P["cls"]) else np.zeros(0, int)
            if len(s):
                bs[j] = box_iou_np(v["b"][j:j + 1], P["box"][s])[0].max()
                det[j] = bs[j] >= 0.5
            if len(P["box"]):
                ba[j] = box_iou_np(v["b"][j:j + 1], P["box"])[0].max()
        W, H = v["w"], v["h"]
        small = a < SMALL_A
        for j in range(n):
            if not small[j]:
                continue
            d = dict(stem=stem, gt=j, cls=int(v["c"][j]), name=NAMES[int(v["c"][j])],
                     area=float(a[j]), diag=float(dg[j]), sqrt=float(np.sqrt(a[j])),
                     in_sqrt=float(np.sqrt(a[j]) * 1280.0 / max(W, H)),
                     detected=bool(det[j]), same_iou=float(bs[j]), any_iou=float(ba[j]),
                     img_w=W, img_h=H)
            # 边界
            m = min(v["b"][j, 0], v["b"][j, 1], W - v["b"][j, 2], H - v["b"][j, 3])
            d["border_margin"] = float(m)
            d["border_norm"] = float(m / min(W, H))
            d["truncated"] = bool(m <= 1.0)
            # 全局密度
            d["img_gt"] = n
            d["img_small"] = int(small.sum())
            d["img_medium"] = int(((a >= SMALL_A) & (a < 9216)).sum())
            d["img_large"] = int((a >= 9216).sum())
            # 多半径局部密度
            for r in RADII:
                mask = (dist[j] <= r * dg[j]); mask[j] = False
                idx = np.where(mask)[0]
                d[f"n_r{r}"] = int(len(idx))
                d[f"n_small_r{r}"] = int(small[idx].sum())
                d[f"n_same_r{r}"] = int((v["c"][idx] == v["c"][j]).sum())
                d[f"n_other_r{r}"] = int((v["c"][idx] != v["c"][j]).sum())
                d[f"n_det_same_r{r}"] = int(det[idx][v["c"][idx] == v["c"][j]].sum()) if len(idx) else 0
                d[f"n_det_other_r{r}"] = int(det[idx][v["c"][idx] != v["c"][j]].sum()) if len(idx) else 0
                if len(idx):
                    d[f"near_gt_r{r}"] = float(dist[j][idx].min() / dg[j])
                    sm = idx[small[idx]]
                    d[f"near_small_r{r}"] = float(dist[j][sm].min() / dg[j]) if len(sm) else None
                    sa = idx[v["c"][idx] == v["c"][j]]
                    d[f"near_same_r{r}"] = float(dist[j][sa].min() / dg[j]) if len(sa) else None
                    oa = idx[v["c"][idx] != v["c"][j]]
                    d[f"near_other_r{r}"] = float(dist[j][oa].min() / dg[j]) if len(oa) else None
                else:
                    for k in ("near_gt", "near_small", "near_same", "near_other"):
                        d[f"{k}_r{r}"] = None
            A[f"{stem}#{j}"] = d
    return A


def main():
    gt, pred = load()
    A = annotate(gt, pred)
    rec = json.loads((OUT / "_v2_records.json").read_text(encoding="utf-8"))["records"]
    attr = {r["key"]: r for r in json.loads((OUT / "_v2_attribute.json").read_text(encoding="utf-8"))}
    E1 = [k for k, r in attr.items() if r["cause"] in ("SCENE_DIFFICULTY", "WEAK_FEATURE_EVIDENCE")]
    zero = [k for k, r in A.items() if r["any_iou"] <= EPS0]
    print(f"small GT={len(A)}  任意类别 IoU==0={len(zero)}  E1 家族={len(E1)}")

    # ---------- §2 matched controls ----------
    print("\n=== §2 matched controls ===")
    ctrl = {}
    n_A = n_A2 = n_B = n_none = 0
    for k in E1:
        t = A[k]
        cand = [j for j, u in A.items()
                if u["stem"] == t["stem"] and j != k and u["detected"]
                and 0.67 <= (u["in_sqrt"] / (t["in_sqrt"] + EPS0)) <= 1.5]
        sc = [j for j in cand if A[j]["cls"] == t["cls"]]
        if sc:
            ctrl[k] = (min(sc, key=lambda j: abs(A[j]["in_sqrt"] - t["in_sqrt"])), "A_same_img_same_cls")
            n_A += 1
        elif cand:
            ctrl[k] = (min(cand, key=lambda j: abs(A[j]["in_sqrt"] - t["in_sqrt"])), "A_same_img_diff_cls")
            n_A2 += 1
        else:
            x = [j for j, u in A.items() if u["cls"] == t["cls"] and u["detected"]
                 and u["img_w"] == t["img_w"] and 0.8 <= (u["in_sqrt"] / (t["in_sqrt"] + EPS0)) <= 1.25]
            if x:
                ctrl[k] = (min(x, key=lambda j: abs(A[j]["in_sqrt"] - t["in_sqrt"])), "B_cross_img")
                n_B += 1
            else:
                n_none += 1
    print(f"  同图同类 control = {n_A}")
    print(f"  同图异类 control = {n_A2}")
    print(f"  跨图 control     = {n_B}")
    print(f"  无可靠 control   = {n_none}")

    # ---------- §3/§4 density + competition ----------
    print("\n=== §4 competition ===")
    comp = Counter()
    for k in E1:
        t = A[k]
        if t["n_det_same_r3"] > 0:
            comp["Competition-A same-class detected nearby"] += 1
        elif t["n_det_other_r3"] > 0:
            comp["Competition-B other-class detected nearby"] += 1
        elif t["n_r3"] > 0:
            comp["Competition-C neighbors all missed"] += 1
        else:
            comp["Competition-D isolated (no neighbor)"] += 1
    for c, v in comp.most_common():
        print(f"  {c:<44} {v:3d}  ({v/len(E1):.1%})")

    print("\n=== §3 局部密度（r=1/2/3/5 × GT 对角），missed 38 vs matched controls ===")
    print(f"  {'':<26}{'38 missed 中位':>16}{'control 中位':>14}")
    for r in RADII:
        for f in ("n_r", "n_same_r", "n_other_r"):
            a = [A[k][f + str(r)] for k in E1]
            b = [A[c[0]][f + str(r)] for c in ctrl.values()]
            print(f"  {f+str(r):<26}{np.median(a):>16.1f}{np.median(b):>14.1f}")

    # ---------- §5 density-stratified miss ----------
    print("\n=== §5 small miss 率是否随局部实例密度单调上升（全部 371 small GT）===")
    print(f"  {'r=1 邻居数':<14}{'n':>6}{'any-zero-overlap 率':>22}")
    for lo, hi, lab in ((0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 99, "4+")):
        g = [u for u in A.values() if lo <= u["n_r1"] <= hi]
        if not g:
            continue
        z = sum(1 for u in g if u["any_iou"] <= EPS0)
        print(f"  {lab:<14}{len(g):>6}{z/len(g):>22.1%}")
    print("  --- 同样按 r=3 ---")
    for lo, hi, lab in ((0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 99, "6+")):
        g = [u for u in A.values() if lo <= u["n_r3"] <= hi]
        if not g:
            continue
        z = sum(1 for u in g if u["any_iou"] <= EPS0)
        print(f"  {lab:<14}{len(g):>6}{z/len(g):>22.1%}")

    # ---------- §6 border ----------
    print("\n=== §6 border analysis ===")
    bins = [(0, .05, "<0.05"), (.05, .10, "0.05-0.10"), (.10, .20, "0.10-0.20"), (.20, 9, ">0.20")]
    print(f"  {'border_norm':<14}{'38 missed':>12}{'占38':>8}{'control':>10}{'占ctrl':>8}{'全体small zero率':>18}")
    for lo, hi, lab in bins:
        a = [k for k in E1 if lo <= A[k]["border_norm"] < hi]
        b = [c[0] for c in ctrl.values() if lo <= A[c[0]]["border_norm"] < hi]
        g = [u for u in A.values() if lo <= u["border_norm"] < hi]
        z = sum(1 for u in g if u["any_iou"] <= EPS0) / max(1, len(g))
        print(f"  {lab:<14}{len(a):>12}{len(a)/len(E1):>8.1%}{len(b):>10}{len(b)/max(1,len(ctrl)):>8.1%}{z:>18.1%}")
    print(f"  截断(margin<=1px) 的 missed: {sum(1 for k in E1 if A[k]['truncated'])}/{len(E1)}")

    # ---------- §7 matrix ----------
    print("\n=== §7 density × same-class-success 矩阵（对 70 个 zero-overlap）===")
    dens = [A[k]["n_r1"] for k in zero]
    thr = float(np.median(dens))
    print(f"  dense 阈值 = r=1 邻居数 >= {thr:.0f}（zero-overlap 集合的中位）")
    M = Counter()
    for k in zero:
        hi = A[k]["n_r1"] >= thr
        suc = A[k]["n_det_same_r3"] > 0
        M[("high" if hi else "low", "success" if suc else "no_success")] += 1
    for a_ in ("high", "low"):
        for b_ in ("success", "no_success"):
            print(f"    A/B/C/D  density={a_:<5} same-class-success={b_:<12} {M[(a_,b_)]:>3}")

    # ---------- §8 animal vs person, density controlled ----------
    print("\n=== §8 animal vs person（叠加密度控制）===")
    AP = [u for u in A.values() if u["cls"] in (0, 2)]
    print(f"  {'密度层(r=1)':<14}{'animal n':>9}{'animal miss':>13}{'person n':>9}{'person miss':>13}")
    for lo, hi, lab in ((0, 0, "0"), (1, 1, "1"), (2, 99, "2+")):
        aa = [u for u in AP if u["cls"] == 2 and lo <= u["n_r1"] <= hi]
        pp = [u for u in AP if u["cls"] == 0 and lo <= u["n_r1"] <= hi]
        ra = sum(1 for u in aa if u["any_iou"] <= EPS0) / max(1, len(aa))
        rp = sum(1 for u in pp if u["any_iou"] <= EPS0) / max(1, len(pp))
        print(f"  {lab:<14}{len(aa):>9}{ra:>13.1%}{len(pp):>9}{rp:>13.1%}")
    print(f"  {'border_norm<0.10':<14}", end="")
    aa = [u for u in AP if u["cls"] == 2 and u["border_norm"] < 0.10]
    pp = [u for u in AP if u["cls"] == 0 and u["border_norm"] < 0.10]
    print(f"{len(aa):>9}{sum(1 for u in aa if u['any_iou']<=EPS0)/max(1,len(aa)):>13.1%}"
          f"{len(pp):>9}{sum(1 for u in pp if u['any_iou']<=EPS0)/max(1,len(pp)):>13.1%}")

    (OUT / "_v3_analysis.json").write_text(json.dumps(
        dict(E1=E1, controls={k: list(v) for k, v in ctrl.items()},
             competition=dict(comp), annotated=A), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {OUT/'_v3_analysis.json'}")
    return A, E1, ctrl, comp


if __name__ == "__main__":
    main()
