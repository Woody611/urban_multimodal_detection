"""_f1_vs_dprime_boot.py — D′ vs F1 官方口径逐项对比 + **配对 image-level bootstrap**（只读）。

约束（与 _e1_vs_dprime_diag.py 一致）：
  · 官方 evaluator（scripts/official_eval.py 的常量与函数**直接 import**），cap=100，
    同一批 400 个 val prediction TXT，同一 val split；
  · 新增推理 = 0；训练 = 0；只读预测 TXT，不写任何数据/模型/提交；
  · 判据 metric 一律 METRIC_A（固定 12 类分母），与官方一致。

本脚本在 diag 之上补三件事：
  ① 配对 bootstrap（按图像重采样，两个模型用**同一组**重采样索引）→ Δ 的 CI 与 P(Δ>0)；
  ② 尺度桶 Δ 的**重归一化**（只对桶内有 GT 的类求均值），因为原始 12 类均值被零稀释；
  ③ 检测级 vs 定位级拆分（ΔmAP50 / ΔmAP75）+ min_conf 扫描 + TP/FP/FN 计数，
     用于把"类别 / 尺度 / IoU 区间 / prediction distribution"四个假设各自证伪或证实。

快速求值器对 official_eval 的复刻已在 main() 里用 evaluate() 做数值断言（不一致即 abort）。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from official_eval import (  # noqa: E402
    EPS, IOU_THRESHOLDS, MAX_BOXES_PER_IMAGE, NC, apply_max_boxes, box_iou_np,
    evaluate, interpolate_ap, load_split, norm_xywh_to_xyxy, read_pred_txt,
)

IMAGES = ROOT / "data/processed/rgbid_split_train/images/val/visible"
LABELS = ROOT / "data/processed/rgbid_split_train/labels/val/visible"
MODELS = {
    "Dp": ROOT / "diagnostic/sepstem_clahe/best_full/results",
    "F1": ROOT / "diagnostic/f1_native_small_replay/infer_val/results",
}
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
BUCKETS = (("all", 0.0, 1e18), ("small", 0.0, 1024.0),
           ("medium", 1024.0, 9216.0), ("large", 9216.0, 1e18))
SEED, B_BOOT = 20260928, 400
CONF_SWEEP = (0.0, 0.001, 0.01, 0.05, 0.10, 0.25)


# ------------------------------------------------------------------ 快速 AP
def ap_fast(recall, precision, n_points=101):
    """等价于 official_eval.interpolate_ap(tail='zero')：101 点，p_interp=max{prec: rec>=r}。

    recall[0.5:] 单调不减 ⇒ {k: rec_k >= r} 是后缀 ⇒ 用后缀 max + searchsorted 一次算完。
    数值等价性在 main() 里逐类断言。
    """
    if recall.size == 0:
        return 0.0
    x = np.linspace(0.0, 1.0, n_points)
    sufmax = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, x, side="left")
    q = np.where(idx < sufmax.size, sufmax[np.minimum(idx, sufmax.size - 1)], 0.0)
    return float(q.mean())


# ------------------------------------------------------------------ 装载
def build_units(per_image, results_dir, lo, hi, max_boxes=MAX_BOXES_PER_IMAGE):
    """每图一个 unit：GT（按 native 面积桶过滤）与预测（**不过滤**，与 diag 的 scale_ap 同语义）。

    预测的 IoU row = 与该图**同类** GT 的 IoU 降序 [(iou, gt_idx)]，纯 Python 小数组，
    避免 bootstrap 里反复调用 numpy 小算子。
    """
    units = []
    for _img_id, stem, w, h, gtb, gtc in per_image:
        gts = {c: [] for c in range(NC)}
        if gtb is not None and len(gtb):
            for b, c in zip(gtb, gtc):
                if lo <= float((b[2] - b[0]) * (b[3] - b[1])) < hi:
                    gts[int(c)].append(b)
        npg = {c: (np.stack(v) if v else None) for c, v in gts.items()}
        preds = {c: [] for c in range(NC)}
        _, pr, _ = read_pred_txt(Path(results_dir) / f"{stem}.txt")
        pr = apply_max_boxes(pr, max_boxes)
        if len(pr):
            boxes, cls = norm_xywh_to_xyxy(pr, w, h)
            for i in range(len(boxes)):
                c = int(cls[i])
                g = npg[c]
                if g is None:
                    row = ()
                else:
                    ious = box_iou_np(boxes[i][None, :], g)[0]
                    row = tuple((float(ious[j]), int(j))
                                for j in np.argsort(-ious))
                preds[c].append((float(pr[i, 5]), row))
        units.append((gts, preds))
    return units


def eval_units(units, unit_list, thresholds=IOU_THRESHOLDS, min_conf=0.0):
    """复刻 official_eval.evaluate（METRIC_A），但接受"单位重采样列表"。"""
    npos = np.zeros(NC, int)
    pool = {c: [] for c in range(NC)}
    for upos, ui in enumerate(unit_list):
        gts, preds = units[ui]
        for c in range(NC):
            npos[c] += len(gts[c])
            for conf, row in preds[c]:
                if conf >= min_conf:
                    pool[c].append((conf, upos, row))
    aps = np.zeros((NC, len(thresholds)))
    rmax = np.zeros(NC)
    npred = np.zeros(NC, int)
    ntp = np.zeros(NC)
    for ci in range(NC):
        pr_c = sorted(pool[ci], key=lambda z: -z[0])       # 稳定排序，与官方同序
        npred[ci] = len(pr_c)
        if npos[ci] == 0:
            continue
        for ti, t in enumerate(thresholds):
            used, tp = {}, np.zeros(len(pr_c))
            for k, (_conf, upos, row) in enumerate(pr_c):
                u = used.get(upos)
                if u is None:                              # 该 unit 该类 GT 的占用位
                    u = used[upos] = [False] * len(units[unit_list[upos]][0][ci])
                hit = False
                for iou, gi in row:                        # 降序扫描，首个未占用者即最大 IoU
                    if u[gi]:
                        continue
                    if iou >= t:
                        u[gi], hit = True, True
                    break
                if hit:
                    tp[k] = 1
            ctp = np.cumsum(tp)
            cfp = np.arange(1, len(pr_c) + 1) - ctp
            aps[ci, ti] = ap_fast(ctp / npos[ci],
                                  ctp / np.maximum(ctp + cfp, EPS))
            if ti == 0:
                rmax[ci] = float(ctp[-1] / npos[ci])
                ntp[ci] = float(ctp[-1])
    per_iou = aps.mean(0)                                  # METRIC_A：固定 12 类
    return dict(per_iou=per_iou, per_class=aps, npos=npos, npred=npred,
                rmax=rmax, ntp=ntp, mAP50=float(per_iou[0]),
                mAP75=float(per_iou[5]), mAP5095=float(per_iou.mean()))


def renorm(aps, npos, ix=slice(None)):
    """只对桶内/重采样内**有 GT 的类**求均值（抵消 12 类分母的零稀释）。"""
    has = npos > 0
    return float(aps[has, ix].mean()) if has.any() else float("nan")


def seg(per_iou):
    return (float(per_iou[:3].mean()), float(per_iou[3:6].mean()),
            float(per_iou[6:].mean()))


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    per_image, stats = load_split(IMAGES, LABELS)
    print("=" * 108)
    print("D′ vs F1 —— 同一官方 evaluator / 同一 400 TXT / cap=100 / 不重新推理 / METRIC_A")
    print(f"  images={stats['images']}  corrupt={stats['corrupt']}  有效图={len(per_image)}  GT={stats['gt']}")
    print("=" * 108)

    U, OFF = {}, {}
    for k, d in MODELS.items():
        U[k] = build_units(per_image, d, 0.0, 1e18)

    # ---- 断言：快速求值器 == official_eval.evaluate ----
    B = {}
    for k, d in MODELS.items():
        off = evaluate(per_image, d, max_boxes=MAX_BOXES_PER_IMAGE)
        fast = eval_units(U[k], list(range(len(per_image))))
        for name, a, b in (("mAP50-95", off["mAP50-95"], fast["mAP5095"]),
                           ("mAP50", off["mAP50"], fast["mAP50"]),
                           ("mAP75", off["mAP75"], fast["mAP75"])):
            assert abs(a - b) < 2e-6, f"[{k}] {name} 复刻失败 off={a} fast={b}"
        for c in range(NC):
            if not np.isnan(off["per_class_ap50_95"][c]):
                assert abs(off["per_class_ap50_95"][c] - fast["per_class"][c].mean()) < 2e-6, \
                    f"[{k}] 类{NAMES[c]} AP 复刻失败"
        B[k] = fast
        print(f"  [{k}] 复刻断言 PASS   mAP50-95={fast['mAP5095']:.5f}  "
              f"mAP50={fast['mAP50']:.5f}  mAP75={fast['mAP75']:.5f}  preds={int(fast['npred'].sum())}")

    D, E = B["Dp"], B["F1"]
    print(f"\n  Δ(F1 − D′) 官方 mAP50-95 = {E['mAP5095'] - D['mAP5095']:+.5f}"
          f"   mAP50 = {E['mAP50'] - D['mAP50']:+.5f}   mAP75 = {E['mAP75'] - D['mAP75']:+.5f}")

    # ---------------- ① 配对 bootstrap ----------------
    rng = np.random.default_rng(SEED)
    N = len(per_image)
    draws = rng.integers(0, N, size=(B_BOOT, N))
    boot = dict(map=np.zeros(B_BOOT), map50=np.zeros(B_BOOT), map75=np.zeros(B_BOOT),
                low=np.zeros(B_BOOT), mid=np.zeros(B_BOOT), high=np.zeros(B_BOOT),
                per_class=np.zeros((B_BOOT, NC)))
    boots = {}
    for lab, lo, hi in BUCKETS:
        if lab != "all":
            for k, d in MODELS.items():
                U[k + "|" + lab] = build_units(per_image, d, lo, hi)
    for b in range(B_BOOT):
        ul = draws[b].tolist()
        pi = {}
        for k in MODELS:
            r = eval_units(U[k], ul)
            pi[k] = r["per_iou"]
            if k == "F1":
                boot["per_class"][b] = r["per_class"].mean(1) - pd_  # Δ = F1 − D′
            else:
                pd_ = r["per_class"].mean(1)
        d = pi["F1"] - pi["Dp"]
        boot["map"][b] = d.mean(); boot["map50"][b] = d[0]; boot["map75"][b] = d[5]
        boot["low"][b], boot["mid"][b], boot["high"][b] = seg(d)
        if b == 0:
            boots["_t"] = time.time() - t0
    # 尺度桶 bootstrap（同一组重采样索引，B 与主循环一致）
    boot["bucket"] = {}
    for k2 in [b[0] for b in BUCKETS if b[0] != "all"]:
        boot["bucket"][k2] = np.zeros(B_BOOT)
    for b in range(B_BOOT):
        ul = draws[b].tolist()
        for k2 in boot["bucket"]:
            a = eval_units(U["Dp|" + k2], ul)["per_iou"].mean()
            e = eval_units(U["F1|" + k2], ul)["per_iou"].mean()
            boot["bucket"][k2][b] = e - a

    def ci(x, label, pt=None, dec=5):
        lo, hi = np.percentile(x, [2.5, 97.5])
        p = float((x > 0).mean())
        pt = "" if pt is None else f"  点估={pt:+.5f}"
        print(f"  {label:<26}{np.mean(x):>+10.5f}[{lo:+.5f}, {hi:+.5f}]"
              f"{'  含 0' if lo < 0 < hi else '  不含 0':<8}{p:>7.3f}{pt}")

    print("\n" + "=" * 108)
    print(f"① 配对 image-level bootstrap（B={B_BOOT}，同一重采样索引喂两个模型）")
    print("=" * 108)
    print(f"  {'量':<26}{'均值Δ':>10}{'95% CI':>22}{'':<8}{'P(Δ>0)':>7}")
    ci(boot["map"], "Δ mAP50-95", E["mAP5095"] - D["mAP5095"])
    ci(boot["map50"], "Δ mAP50", E["mAP50"] - D["mAP50"])
    ci(boot["map75"], "Δ mAP75", E["mAP75"] - D["mAP75"])
    ci(boot["low"], "Δ IoU .50-.60 段", float((E["per_iou"][:3] - D["per_iou"][:3]).mean()))
    ci(boot["mid"], "Δ IoU .65-.75 段", float((E["per_iou"][3:6] - D["per_iou"][3:6]).mean()))
    ci(boot["high"], "Δ IoU .80-.95 段", float((E["per_iou"][6:] - D["per_iou"][6:]).mean()))
    for k2 in boot["bucket"]:
        a = eval_units(U["Dp|" + k2], list(range(N)))["per_iou"].mean()
        e = eval_units(U["F1|" + k2], list(range(N)))["per_iou"].mean()
        ci(boot["bucket"][k2], f"Δ [{k2}]（12 类分母）", e - a)

    # ---------------- ② 类别 ----------------
    print("\n" + "=" * 108)
    print("② 类别分解：AP50-95 + 配对 bootstrap CI（Δ 是否可与 0 区分）")
    print("=" * 108)
    print(f"  {'class':<13}{'GT n':>6}{'Dp':>9}{'F1':>9}{'Δ':>10}{'95% CI':>20}{'P(Δ>0)':>8}")
    gt_n = {c: int(D["npos"][c]) for c in range(NC)}
    rows = []
    for c in range(NC):
        if gt_n[c] == 0:
            continue
        dd = float(E["per_class"][c].mean() - D["per_class"][c].mean())
        bc = boot["per_class"][:, c]
        lo, hi = np.percentile(bc, [2.5, 97.5])
        rows.append((dd, c, gt_n[c], float(D["per_class"][c].mean()),
                     float(E["per_class"][c].mean()), lo, hi, float((bc > 0).mean())))
    for dd, c, n, a, b, lo, hi, p in sorted(rows):
        print(f"  {NAMES[c]:<13}{n:>6}{a:>9.4f}{b:>9.4f}{dd:>+10.4f}"
              f"  [{lo:+.4f},{hi:+.4f}]{p:>8.3f}")
    net = sum(r[0] for r in rows) / NC
    neg = sum(min(r[0], 0) for r in rows); pos = sum(max(r[0], 0) for r in rows)
    sig = [r for r in rows if not (r[5] < 0 < r[6])]
    print(f"\n  ΣΔ/12 = {net:+.5f}   负向合计 {neg/NC:+.5f} / 正向合计 {pos/NC:+.5f}")
    print(f"  Δ 的 95% CI **不含 0** 的类：{len(sig)}/{len(rows)} → "
          + (", ".join(f"{NAMES[r[1]]}({r[0]:+.4f}, n={r[2]})" for r in sorted(sig))
             if sig else "无"))
    top2 = sum(r[0] for r in sorted(rows)[:2])
    print(f"  最负两类合计 {top2:+.5f} = **{abs(top2/net):.1f}× 净 ΣΔ**（被正向类抵消后净额很小）"
          f"；占**负向总量** {top2/neg:.0%}"
          f"\n  （diag 的 94% 就是 top2/净额，分母是净值 ⇒ 会被放大；负向总量占比才是集中度）")

    # ---------------- ③ 尺度 ----------------
    print("\n" + "=" * 108)
    print("③ 尺度分解（native 面积桶；GT 过滤 / 预测不过滤；两列口径）")
    print("=" * 108)
    print(f"  {'bucket':<9}{'GT n':>6}{'Dp':>9}{'F1':>9}{'Δ':>10}"
          f"{'Δ重归一化':>12}{'含GT类数':>9}")
    for lab, lo, hi in BUCKETS:
        sfx = "" if lab == "all" else "|" + lab
        rd = eval_units(U["Dp" + sfx], list(range(N)))
        re_ = eval_units(U["F1" + sfx], list(range(N)))
        ngt = int(rd["npos"].sum()); ncls = int((rd["npos"] > 0).sum())
        a, b = rd["per_iou"].mean(), re_["per_iou"].mean()
        print(f"  {lab:<9}{ngt:>6}{a:>9.4f}{b:>9.4f}{b-a:>+10.4f}"
              f"{renorm(re_['per_class'], re_['npos']) - renorm(rd['per_class'], rd['npos']):>+12.4f}"
              f"{ncls:>9}")
    print("  注：'Δ重归一化' 只对桶内有 GT 的类求均值，是可比口径；上行 Δ 是官方 12 类分母（零稀释）。")

    # ---------------- ④ IoU 区间 ----------------
    print("\n" + "=" * 108)
    print("④ IoU 区间分解（官方 10 个阈值；bootstrap CI 见 ①）")
    print("=" * 108)
    print(f"  {'IoU':<7}{'Dp':>10}{'F1':>10}{'Δ':>10}{'  累计段':<10}")
    for t, a, b in zip(IOU_THRESHOLDS, D["per_iou"], E["per_iou"]):
        print(f"  {t:<7.2f}{a:>10.4f}{b:>10.4f}{b-a:>+10.4f}")
    dl, dm, dh = seg(E["per_iou"] - D["per_iou"])
    print(f"  段均值 Δ：低(.50-.60) {dl:+.5f} | 中(.65-.75) {dm:+.5f} | 高(.80-.95) {dh:+.5f}")

    # ---------------- ⑤ prediction distribution ----------------
    print("\n" + "=" * 108)
    print("⑤ prediction distribution：计数 / 置信度 / 尺寸 / min_conf 稳健性")
    print("=" * 108)
    for k in ("Dp", "F1"):
        r = B[k]
        print(f"  [{k}] 预测总数 {int(r['npred'].sum())}  GT {int(r['npos'].sum())}  "
              f"TP@IoU.50 {int(r['ntp'].sum())}  最大召回 {r['ntp'].sum()/r['npos'].sum():.4f}")
    print(f"\n  {'class':<13}{'GT':>5}{'pred Dp':>9}{'pred F1':>9}{'Δpred':>7}"
          f"{'Rmax Dp':>9}{'Rmax F1':>9}{'ΔRmax':>8}")
    for c in range(NC):
        if gt_n[c] == 0:
            continue
        print(f"  {NAMES[c]:<13}{gt_n[c]:>5}{int(D['npred'][c]):>9}{int(E['npred'][c]):>9}"
              f"{int(E['npred'][c]-D['npred'][c]):>+7}{D['rmax'][c]:>9.4f}"
              f"{E['rmax'][c]:>9.4f}{E['rmax'][c]-D['rmax'][c]:>+8.4f}")
    print(f"\n  {'min_conf':<10}{'Dp':>10}{'F1':>10}{'Δ':>10}{'Δ mAP50':>11}{'preds Dp/F1':>16}")
    for mc in CONF_SWEEP:
        a = eval_units(U["Dp"], list(range(N)), min_conf=mc)
        e = eval_units(U["F1"], list(range(N)), min_conf=mc)
        print(f"  {mc:<10.4f}{a['mAP5095']:>10.5f}{e['mAP5095']:>10.5f}"
              f"{e['mAP5095']-a['mAP5095']:>+10.5f}"
              f"{e['mAP50']-a['mAP50']:>+11.5f}"
              f"{int(a['npred'].sum()):>9}/{int(e['npred'].sum()):<6}")
    print("\n  运行耗时 %.1f s" % (time.time() - t0))

    out = dict(delta=E["mAP5095"] - D["mAP5095"],
               bootstrap=dict(B=B_BOOT, seed=SEED,
                              map=[float(np.mean(boot["map"])),
                                   [float(x) for x in np.percentile(boot["map"], [2.5, 97.5])],
                                   float((boot["map"] > 0).mean())],
                              map50=[float(np.mean(boot["map50"])),
                                     [float(x) for x in np.percentile(boot["map50"], [2.5, 97.5])]],
                              map75=[float(np.mean(boot["map75"])),
                                     [float(x) for x in np.percentile(boot["map75"], [2.5, 97.5])]],
                              low=[float(np.mean(boot["low"])),
                                   [float(x) for x in np.percentile(boot["low"], [2.5, 97.5])]],
                              mid=[float(np.mean(boot["mid"])),
                                   [float(x) for x in np.percentile(boot["mid"], [2.5, 97.5])]],
                              high=[float(np.mean(boot["high"])),
                                    [float(x) for x in np.percentile(boot["high"], [2.5, 97.5])]],
                              bucket={k2: [float(np.mean(v)),
                                           [float(x) for x in np.percentile(v, [2.5, 97.5])]]
                                      for k2, v in boot["bucket"].items()}),
               per_class=[[NAMES[c], gt_n[c], float(D["per_class"][c].mean()),
                           float(E["per_class"][c].mean()),
                           float(E["per_class"][c].mean() - D["per_class"][c].mean()),
                           [float(x) for x in np.percentile(boot["per_class"][:, c], [2.5, 97.5])]]
                          for _, c, *_ in rows])
    (ROOT / "diagnostic/f1_native_small_replay/_f1_vs_dprime_boot.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
