"""_v2_matched.py — V2 阶段 5（同图同类配对）/ 6（animal vs person 三层控制）/ 7（弱候选）。

只读。复用 _v2_records.py 已产出的 _v2_records.json 与同一套 official_eval 口径。
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = Path(__file__).resolve().parent

from _v2_records import (  # noqa: E402
    CONF_FLOOR, EPS0, IMAGES, LABELS, NAMES, SMALL_A, Dp, iter_split,
)
from official_eval import (  # noqa: E402
    MAX_BOXES_PER_IMAGE, apply_max_boxes, box_iou_np, norm_xywh_to_xyxy, read_pred_txt,
)


def load_all():
    val = {}
    for stem, w, h, b, c in iter_split(IMAGES, LABELS):
        val[stem] = dict(w=w, h=h, b=b, c=c)
    pred = {}
    for stem, v in val.items():
        _, p, _ = read_pred_txt(Dp / f"{stem}.txt")
        p = apply_max_boxes(p, MAX_BOXES_PER_IMAGE)
        if len(p):
            pb, pc = norm_xywh_to_xyxy(p, v["w"], v["h"])
            pred[stem] = dict(box=pb, cls=pc, conf=p[:, 5].astype(np.float32))
        else:
            pred[stem] = dict(box=np.zeros((0, 4), np.float32), cls=np.zeros(0, int),
                              conf=np.zeros(0, np.float32))
    return val, pred


def gt_status(v, P):
    """每个 GT：best same-class IoU / best any IoU / detected(>=0.5) / conf"""
    n = len(v["b"])
    bs = np.zeros(n, np.float32); ba = np.zeros(n, np.float32)
    cs = np.zeros(n, np.float32)
    for j in range(n):
        same = np.where(P["cls"] == v["c"][j])[0] if len(P["cls"]) else np.zeros(0, int)
        if len(same):
            i = box_iou_np(v["b"][j:j + 1], P["box"][same])[0]
            k = int(np.argmax(i)); bs[j] = i[k]; cs[j] = P["conf"][same[k]]
        if len(P["box"]):
            ba[j] = box_iou_np(v["b"][j:j + 1], P["box"])[0].max()
    a = np.clip(v["b"][:, 2] - v["b"][:, 0], 0, None) * np.clip(v["b"][:, 3] - v["b"][:, 1], 0, None)
    return bs, ba, cs, a


def main():
    val, pred = load_all()
    print("=" * 78)
    print("阶段 5 —— 同图 / 同类 / 相似尺寸 的 missed vs detected 配对")
    print("=" * 78)
    pairs = []
    for stem, v in val.items():
        if v["b"] is None or len(v["b"]) == 0:
            continue
        P = pred[stem]
        bs, ba, cs, a = gt_status(v, P)
        sc = 1280.0 / max(v["w"], v["h"])
        small = np.where(a < SMALL_A)[0]
        missed = [j for j in small if ba[j] <= EPS0]
        det = [j for j in small if bs[j] >= 0.5]
        for j in missed:
            sj = np.sqrt(a[j])
            for k in det:
                if v["c"][k] != v["c"][j]:
                    continue
                r = np.sqrt(a[k]) / (sj + EPS0)
                if not (0.67 <= r <= 1.5):
                    continue
                pairs.append(dict(stem=stem, cls=int(v["c"][j]), name=NAMES[int(v["c"][j])],
                                  m_sqrt=sj, d_sqrt=np.sqrt(a[k]), ratio=r,
                                  m_area_ratio_in=None, m_any=float(ba[j]), d_any=float(ba[k]),
                                  d_conf=float(cs[k]),
                                  m_input=float(sj * sc), d_input=float(np.sqrt(a[k]) * sc),
                                  m_border=float(min(v["b"][j, 0], v["b"][j, 1], v["w"] - v["b"][j, 2],
                                                     v["h"] - v["b"][j, 3])),
                                  d_border=float(min(v["b"][k, 0], v["b"][k, 1], v["w"] - v["b"][k, 2],
                                                     v["h"] - v["b"][k, 3]))))
    print(f"  配对总数 = {len(pairs)}（{len(set(p['stem'] for p in pairs))} 张图）")
    byc = Counter(p["name"] for p in pairs)
    print(f"  按类别: {dict(byc.most_common())}")
    if pairs:
        r = np.array([p["ratio"] for p in pairs])
        print(f"  size ratio 中位 = {np.median(r):.3f}（配对已控制在 0.67-1.5）")
        print(f"  missed 的 best-any IoU 全为 0（构造使然）；detected 侧 best-any IoU 中位 = "
              f"{np.median([p['d_any'] for p in pairs]):.3f}，conf 中位 = {np.median([p['d_conf'] for p in pairs]):.3f}")
        print(f"  边框距离(native px) 中位: missed={np.median([p['m_border'] for p in pairs]):.1f} "
              f"vs detected={np.median([p['d_border'] for p in pairs]):.1f}")
        # 每张图里 miss/det 的尺寸是否可比
        print(f"  输入空间 √area 中位: missed={np.median([p['m_input'] for p in pairs]):.2f} "
              f"vs detected={np.median([p['d_input'] for p in pairs]):.2f}")

    # ---------- 阶段 6：animal vs person ----------
    print("\n" + "=" * 78)
    print("阶段 6 —— animal vs person（三层控制）")
    print("=" * 78)
    R = {r["image_id"] + f"#{r['gt_id']}": r for r in
         json.loads((OUT / "_v2_records.json").read_text(encoding="utf-8"))["records"]}
    rows = []
    for stem, v in val.items():
        if v["b"] is None or len(v["b"]) == 0:
            continue
        P = pred[stem]
        bs, ba, cs, a = gt_status(v, P)
        sc = 1280.0 / max(v["w"], v["h"])
        for j in range(len(v["b"])):
            if a[j] >= SMALL_A or int(v["c"][j]) not in (0, 2):
                continue
            rows.append(dict(stem=stem, cls=int(v["c"][j]), name=NAMES[int(v["c"][j])],
                             sqrt=float(np.sqrt(a[j])), in_sqrt=float(np.sqrt(a[j]) * sc),
                             miss=bool(ba[j] <= EPS0), det=bool(bs[j] >= 0.5),
                             res=f"{v['w']}x{v['h']}", any_iou=float(ba[j])))

    def rate(g):
        return (sum(r["miss"] for r in g) / len(g)) if g else float("nan")

    A = [r for r in rows if r["cls"] == 2]; Pn = [r for r in rows if r["cls"] == 0]
    print(f"  未控制     : animal n={len(A)} miss={rate(A):.1%} | person n={len(Pn)} miss={rate(Pn):.1%}")
    # Level 1: 真正的尺寸匹配 —— **按尺寸分箱**，只在两类的共同尺寸区间内比较
    lo = max(np.percentile([r["in_sqrt"] for r in A], 5),
             np.percentile([r["in_sqrt"] for r in Pn], 5))
    hi = min(np.percentile([r["in_sqrt"] for r in A], 95),
             np.percentile([r["in_sqrt"] for r in Pn], 95))
    print(f"  L1 共同尺寸区间 = [{lo:.2f}, {hi:.2f}] px   （两类 P5–P95 的交集）")
    edges = np.linspace(lo, hi, 4)
    print(f"  {'size bin (px)':<18}{'animal n':>9}{'animal miss':>13}{'person n':>9}{'person miss':>13}")
    for i in range(len(edges) - 1):
        a_ = [r for r in A if edges[i] <= r["in_sqrt"] < edges[i + 1]]
        p_ = [r for r in Pn if edges[i] <= r["in_sqrt"] < edges[i + 1]]
        if not a_ and not p_:
            continue
        print(f"  {f'{edges[i]:.1f}-{edges[i+1]:.1f}':<18}{len(a_):>9}{rate(a_):>13.1%}"
              f"{len(p_):>9}{rate(p_):>13.1%}")
    a_in = [r for r in A if lo <= r["in_sqrt"] <= hi]
    p_in = [r for r in Pn if lo <= r["in_sqrt"] <= hi]
    print(f"  区间内合计        animal n={len(a_in)} miss={rate(a_in):.1%} | "
          f"person n={len(p_in)} miss={rate(p_in):.1%}")
    # Level 2: + resolution matched
    for res in sorted(set(r["res"] for r in rows)):
        a2 = [r for r in A if r["res"] == res]; p2 = [r for r in Pn if r["res"] == res]
        if len(a2) >= 5 and len(p2) >= 5:
            print(f"  L2 分辨率={res:<10}: animal n={len(a2)} miss={rate(a2):.1%} "
                  f"| person n={len(p2)} miss={rate(p2):.1%}")
    # Level 3: same-image pairs
    same_img = 0; a_miss = 0
    for stem in set(r["stem"] for r in A):
        ai = [r for r in A if r["stem"] == stem]; pi = [r for r in Pn if r["stem"] == stem]
        if ai and pi:
            same_img += len(ai); a_miss += sum(r["miss"] for r in ai)
    print(f"  L3 同图对: animal 与 person 共现的图里 animal n={same_img} miss="
          f"{a_miss/max(1,same_img):.1%}")

    # ---------- 阶段 7：弱候选 ----------
    print("\n" + "=" * 78)
    print("阶段 7 —— 弱候选（dump 下限 conf>=0.001）")
    print("=" * 78)
    zero = [r for r in R.values() if r["best_any_class_iou"] <= EPS0]
    print(f"  完全无重叠 small GT = {len(zero)}")
    print(f"  ⚠ predict_rect 的 dump 下限 = --conf 0.001 ⇒ conf>=0.0001 这一档 **无法回答**")
    for t in (0.001, 0.01, 0.05):
        have = sum(1 for r in zero if r.get(f"n_pred_conf_ge_{t}", 0) > 0)
        print(f"    该图内存在 conf>={t} 的任意预测: {have}/{len(zero)} = {have/len(zero):.1%}")
    nosame = sum(1 for r in zero if r["n_pred_same"] == 0)
    print(f"    该图内完全没有**同类**预测: {nosame}/{len(zero)} = {nosame/len(zero):.1%}")
    print(f"    ⇒ 分类：NO_CANDIDATE={nosame}  WEAK_OR_OTHER={len(zero)-nosame}")
    return pairs, rows, zero


if __name__ == "__main__":
    main()
