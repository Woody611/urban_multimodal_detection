"""_sge_analyze.py — 由 replay 产物生成 Table 1–4 与 A/B/C/D 裁决（只读、纯离线）

输入：_sge_replay.npz / _sge_native_meta.json / _sge_replay_meta.json（+ 可选 _sge_supervision.json）
输出：_sge_tables.json + 控制台表格

口径说明（不新造定义）：
  native small  := native area < 1024 px²（sqrt(w·h) < 32），**沿用项目定义**
  native 面积   := 归一化 bbox × **原图** 像素尺寸（.txt 归一化到原图）
  final 面积    := Format 输出空间（1280×1280 输入画布）像素面积
  Exposure-A    := P(某 epoch 该 native GT 至少产生 1 个 final target)
  Exposure-B    := P(某 epoch 至少 1 个 final target 且其 n_pos > 0)  ← 需要 supervision 采样
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent

SMALL_BUCKETS = ["<8", "8-12", "12-18", "18-24", "24-32"]


def bucket(s):
    if s < 8:
        return "<8"
    if s < 12:
        return "8-12"
    if s < 18:
        return "12-18"
    if s < 24:
        return "18-24"
    if s < 32:
        return "24-32"
    if s < 96:
        return "32-96"
    return ">96"


def size_class(area):
    return "small" if area < 1024 else ("medium" if area < 9216 else "large")


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def pct(xs, q):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.percentile(xs, q)) if xs else float("nan")


def load():
    meta = json.loads((OUT / "_sge_native_meta.json").read_text(encoding="utf-8"))
    meta = {int(k): v for k, v in meta.items()}
    z = np.load(OUT / "_sge_replay.npz")
    rep = json.loads((OUT / "_sge_replay_meta.json").read_text(encoding="utf-8"))
    return meta, z, rep


def counters(z, tag, key):
    u = z[f"{tag}__{key}__uid"]
    v = z[f"{tag}__{key}__val"]
    return {int(a): int(b) for a, b in zip(u, v)}


def areas_of(z, tag, key):
    """扁平 (uid, area) -> {uid: np.array}。扁平存储可避免每 uid 一个小 list 对象
    （per-uid list 在 ~11k GT × 100 epoch 时造成 GC/内存压力，实测使 epoch 耗时翻倍）。"""
    u = z[f"{tag}__{key}__uid"]
    flat = z[f"{tag}__{key}__flat"]
    order = np.argsort(u, kind="stable")
    us, fs = u[order], flat[order]
    bounds = np.flatnonzero(np.r_[True, us[1:] != us[:-1], True])
    return {int(us[bounds[i]]): fs[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)}


def main():
    meta, z, rep = load()
    tags = [t for t in ("mosaic_ON", "mosaic_OFF") if f"{t}__inst__uid" in z]
    nrep = {t: rep["summary"][t]["n_replays"] for t in tags}
    print("=" * 100)
    print("D′ SMALL-GT EFFECTIVE EXPOSURE — 分析")
    print("=" * 100)
    for t in tags:
        print(f"  {t}: n_images={rep['summary'][t]['n_images']}  n_replays={nrep[t]}  "
              f"native_GT={rep['summary'][t]['n_native_gt']}  mosaic_p={rep['summary'][t]['mosaic_p']}")

    C = {t: {k: counters(z, t, k) for k in ("inst", "inst_main", "inst_tile", "mosaic", "final",
                                            "ep_inst", "ep_final")} for t in tags}
    FA = {t: areas_of(z, t, "final_area") for t in tags}
    MA = {t: areas_of(z, t, "mosaic_area") for t in tags}

    # ---------- 每 uid 汇总 ----------
    def per_uid(t):
        r = {}
        for u, m in meta.items():
            c = C[t]
            inst = c["inst"].get(u, 0)
            r[u] = dict(
                native_area=m["native_area"], native_sqrt=math.sqrt(max(m["native_area"], 0.0)),
                layer=m["layer"], cls=m["cls"],
                inst_per_ep=inst / nrep[t],
                inst_tile_per_ep=c["inst_tile"].get(u, 0) / nrep[t],
                p_inst_ep=c["ep_inst"].get(u, 0) / nrep[t],
                p_mos_given_inst=(c["mosaic"].get(u, 0) / inst) if inst else float("nan"),
                p_fin_given_inst=(c["final"].get(u, 0) / inst) if inst else float("nan"),
                p_fin_given_mos=((c["final"].get(u, 0) / c["mosaic"][u]) if c["mosaic"].get(u, 0) else float("nan")),
                exposureA=c["ep_final"].get(u, 0) / nrep[t],
                n_final_occ=c["final"].get(u, 0),
                fa=np.asarray(FA[t].get(u, []), dtype=np.float64),
                ma=np.asarray(MA[t].get(u, []), dtype=np.float64),
            )
        return r

    P = {t: per_uid(t) for t in tags}

    # ---------- Table 1：pipeline survival（按 native 尺寸分桶）----------
    def table1(t, buckets):
        rows = []
        for b in buckets:
            us = [u for u, r in P[t].items() if bucket(r["native_sqrt"]) == b]
            if not us:
                continue
            inst = [P[t][u]["inst_per_ep"] for u in us]
            pinst = [P[t][u]["p_inst_ep"] for u in us]
            pmos = [P[t][u]["p_mos_given_inst"] for u in us]
            pfin = [P[t][u]["p_fin_given_inst"] for u in us]
            ea = [P[t][u]["exposureA"] for u in us]
            # final-small = 该 native GT 到达 final 的实例中，final 面积仍 <1024 的比例
            fs = []
            for u in us:
                fa = P[t][u]["fa"]
                if len(fa):
                    fs.append(float((fa < 1024).mean()))
            rows.append(dict(bucket=b, n=len(us),
                             inst_per_ep=float(np.mean(inst)),
                             p_inst_ep=float(np.mean(pinst)),
                             p_mos_given_inst=float(np.nanmean(pmos)),
                             p_fin_given_inst=float(np.nanmean(pfin)),
                             exposureA=float(np.mean(ea)),
                             p_final_small_given_final=med(fs)))
        return rows

    # ---------- Table 3：small vs medium/large ----------
    def table3(t):
        out = {}
        for cls in ("small", "medium", "large"):
            us = [u for u, r in P[t].items() if size_class(r["native_area"]) == cls]
            if not us:
                continue
            out[cls] = dict(
                n=len(us),
                exposureA=float(np.mean([P[t][u]["exposureA"] for u in us])),
                p_fin_given_inst=float(np.nanmean([P[t][u]["p_fin_given_inst"] for u in us])),
                inst_per_ep=float(np.mean([P[t][u]["inst_per_ep"] for u in us])),
                zero_final=float(np.mean([1.0 if P[t][u]["n_final_occ"] == 0 else 0.0 for u in us])),
            )
        return out

    # ---------- Table 4：augmentation contribution ----------
    def table4():
        on, off = "mosaic_ON", "mosaic_OFF"
        if on not in P or off not in P:
            return {}
        res = {}
        for cls in ("small", "medium", "large"):
            uo = [u for u, r in P[on].items() if size_class(r["native_area"]) == cls]
            uf = [u for u, r in P[off].items() if size_class(r["native_area"]) == cls]
            if not uo or not uf:
                continue
            res[cls] = dict(
                n=len(uo),
                inst_per_ep_on=float(np.mean([P[on][u]["inst_per_ep"] for u in uo])),
                inst_per_ep_off=float(np.mean([P[off][u]["inst_per_ep"] for u in uf])),
                mosaic_drop=float(np.nanmean([1 - P[on][u]["p_mos_given_inst"] for u in uo])),
                rp_drop_on=float(np.nanmean([1 - P[on][u]["p_fin_given_mos"] for u in uo])),
                rp_drop_off=float(np.nanmean([1 - P[off][u]["p_fin_given_mos"] for u in uf])),
                exposureA_on=float(np.mean([P[on][u]["exposureA"] for u in uo])),
                exposureA_off=float(np.mean([P[off][u]["exposureA"] for u in uf])),
            )
        return res

    # ---------- 缩放（final / native，working space 归一）----------
    def shrink(t):
        rows = []
        for cls in ("small", "medium", "large"):
            us = [u for u, r in P[t].items() if size_class(r["native_area"]) == cls]
            r_ = []
            for u in us:
                fa = P[t][u]["fa"]
                if not len(fa) or P[t][u]["native_area"] <= 0:
                    continue
                f = 1280.0 / max(meta[u]["ori_h"], meta[u]["ori_w"])
                work = P[t][u]["native_area"] * f * f
                r_.append(math.sqrt(max(float(np.median(fa)), 0.0)) / math.sqrt(max(work, 1e-9)))
            if r_:
                rows.append(dict(cls=cls, n=len(r_), median=med(r_), p10=pct(r_, 10), p90=pct(r_, 90),
                                 lt_half=float(np.mean([x < 0.5 for x in r_])),
                                 half_to_1=float(np.mean([0.5 <= x < 1 for x in r_])),
                                 ge_1=float(np.mean([x >= 1 for x in r_]))))
        return rows

    # ---------- 输出 ----------
    res = {}
    for t in tags:
        print(f"\n{'=' * 100}\nTABLE 1 — Pipeline survival（{t}）\n{'=' * 100}")
        print(f"{'native sqrt':>12} {'n':>6} {'inst/ep':>9} {'P(inst)':>9} {'mos|inst':>9} "
              f"{'fin|inst':>9} {'ExposureA':>10} {'fin→small':>10}")
        t1 = table1(t, SMALL_BUCKETS + ["32-96", ">96"])
        for r in t1:
            print(f"{r['bucket']:>12} {r['n']:>6} {r['inst_per_ep']:>9.4f} {r['p_inst_ep']:>9.4f} "
                  f"{r['p_mos_given_inst']:>9.4f} {r['p_fin_given_inst']:>9.4f} "
                  f"{r['exposureA']:>10.4f} {r['p_final_small_given_final']:>10.4f}")
        print(f"\nTABLE 3 — small vs medium/large（{t}）")
        t3 = table3(t)
        print(f"{'class':>8} {'n':>6} {'ExposureA':>10} {'fin|inst':>9} {'inst/ep':>9} {'zero-final%':>12}")
        for k, v in t3.items():
            print(f"{k:>8} {v['n']:>6} {v['exposureA']:>10.4f} {v['p_fin_given_inst']:>9.4f} "
                  f"{v['inst_per_ep']:>9.4f} {100 * v['zero_final']:>11.2f}%")
        if "small" in t3 and "medium" in t3 and "large" in t3:
            s, m, l = t3["small"], t3["medium"], t3["large"]
            print(f"  ratio small/medium ExposureA = {s['exposureA'] / max(m['exposureA'], 1e-9):.4f} ; "
                  f"small/large = {s['exposureA'] / max(l['exposureA'], 1e-9):.4f}")
        print(f"\n  缩放比 final_sqrt / native_work_sqrt（按类）:")
        for r in shrink(t):
            print(f"    {r['cls']:>7} n={r['n']:<6} median={r['median']:.3f} p10={r['p10']:.3f} "
                  f"p90={r['p90']:.3f} <0.5x={100 * r['lt_half']:.1f}% 0.5-1x={100 * r['half_to_1']:.1f}% "
                  f">=1x={100 * r['ge_1']:.1f}%")
        res[t] = dict(table1=t1, table3=t3, shrink=shrink(t))

    print(f"\n{'=' * 100}\nTABLE 4 — Augmentation contribution（mosaic-ON vs OFF，按类）\n{'=' * 100}")
    t4 = table4()
    res["table4"] = t4
    if t4:
        print(f"{'class':>8} {'n':>6} {'inst/ep ON':>11} {'inst/ep OFF':>12} {'mosaic drop':>12} "
              f"{'RP drop ON':>11} {'RP drop OFF':>12} {'ExpA ON':>9} {'ExpA OFF':>9}")
        for k, v in t4.items():
            print(f"{k:>8} {v['n']:>6} {v['inst_per_ep_on']:>11.4f} {v['inst_per_ep_off']:>12.4f} "
                  f"{v['mosaic_drop']:>12.4f} {v['rp_drop_on']:>11.4f} {v['rp_drop_off']:>12.4f} "
                  f"{v['exposureA_on']:>9.4f} {v['exposureA_off']:>9.4f}")

    # ---------- 分层敏感性：native 面积在原图像素上会混淆 HR/LR 两个分辨率层 ----------
    def by_layer(t):
        rows = []
        for lay in ("HR", "LR"):
            for cls in ("small", "medium", "large"):
                us = [u for u, r in P[t].items() if r["layer"] == lay and size_class(r["native_area"]) == cls]
                if not us:
                    continue
                rows.append(dict(layer=lay, cls=cls, n=len(us),
                                 native_sqrt_median=med([P[t][u]["native_sqrt"] for u in us]),
                                 work_sqrt_median=med([math.sqrt(meta[u]["native_area"] *
                                                                 (1280.0 / max(meta[u]["ori_h"], meta[u]["ori_w"])) ** 2)
                                                       for u in us]),
                                 exposureA=float(np.mean([P[t][u]["exposureA"] for u in us])),
                                 p_fin_given_inst=float(np.nanmean([P[t][u]["p_fin_given_inst"] for u in us])),
                                 inst_per_ep=float(np.mean([P[t][u]["inst_per_ep"] for u in us]))))
        return rows

    print("\n" + "=" * 100)
    print("分层敏感性 — HR(原图高>=1000px) vs LR（native 像素面积会混淆两层）")
    print("=" * 100)
    for t in tags:
        print(f"\n  [{t}]")
        print(f"  {'layer':>5} {'class':>7} {'n':>6} {'native_sqrt':>12} {'work_sqrt':>10} "
              f"{'ExposureA':>10} {'fin|inst':>9} {'inst/ep':>8}")
        bl = by_layer(t)
        for r in bl:
            print(f"  {r['layer']:>5} {r['cls']:>7} {r['n']:>6} {r['native_sqrt_median']:>12.1f} "
                  f"{r['work_sqrt_median']:>10.1f} {r['exposureA']:>10.4f} "
                  f"{r['p_fin_given_inst']:>9.4f} {r['inst_per_ep']:>8.4f}")
        res[f"by_layer_{t}"] = bl

    # ---------- §8 Mosaic 作用 / §9 RandomPerspective+scale ----------
    def aug_effect(t):
        rows = []
        for cls in ("small", "medium", "large"):
            us = [u for u, r in P[t].items() if size_class(r["native_area"]) == cls]
            rmos, rfin, mos_shift, fin_shift = [], [], {"smaller": 0, "same": 0, "larger": 0}, {"smaller": 0, "same": 0, "larger": 0}
            nmos = nfin = 0
            for u in us:
                m = meta[u]
                if m["native_area"] <= 0:
                    continue
                f = 1280.0 / max(m["ori_h"], m["ori_w"])
                work_sqrt = math.sqrt(m["native_area"] * f * f)
                if work_sqrt <= 0:
                    continue
                for a in P[t][u]["ma"]:
                    rmos.append(math.sqrt(max(a, 0.0)) / work_sqrt); nmos += 1
                for a in P[t][u]["fa"]:
                    rfin.append(math.sqrt(max(a, 0.0)) / work_sqrt); nfin += 1
                for a in P[t][u]["ma"]:
                    x = math.sqrt(max(a, 0.0)) / work_sqrt
                    mos_shift["smaller" if x < 0.8 else ("same" if x <= 1.25 else "larger")] += 1
                for a in P[t][u]["fa"]:
                    x = math.sqrt(max(a, 0.0)) / work_sqrt
                    fin_shift["smaller" if x < 0.8 else ("same" if x <= 1.25 else "larger")] += 1
            if not rmos and not rfin:
                continue
            rows.append(dict(cls=cls, n_gt=len(us), n_mosaic_occ=nmos, n_final_occ=nfin,
                             mosaic_ratio_median=med(rmos), mosaic_ratio_p10=pct(rmos, 10),
                             mosaic_ratio_p90=pct(rmos, 90),
                             final_ratio_median=med(rfin), final_ratio_p10=pct(rfin, 10),
                             final_ratio_p90=pct(rfin, 90),
                             final_lt_half=float(np.mean([x < 0.5 for x in rfin])) if rfin else float("nan"),
                             mosaic_shift=mos_shift, final_shift=fin_shift))
        return rows

    print("\n" + "=" * 100)
    print("§8/§9 — Mosaic 与 RandomPerspective(+scale) 对尺寸的作用")
    print("（ratio = sqrt(area) / sqrt(native 在 1280 长边空间下的面积)；1.0 = 未缩放）")
    print("=" * 100)
    for t in tags:
        print(f"\n  [{t}]")
        print(f"  {'class':>7} {'n_gt':>6} {'mos_occ':>8} {'mos ratio med':>14} {'p10':>7} {'p90':>7} "
              f"{'fin ratio med':>14} {'p10':>7} {'p90':>7} {'fin<0.5x':>9}")
        ae = aug_effect(t)
        for r in ae:
            print(f"  {r['cls']:>7} {r['n_gt']:>6} {r['n_mosaic_occ']:>8} {r['mosaic_ratio_median']:>14.3f} "
                  f"{r['mosaic_ratio_p10']:>7.3f} {r['mosaic_ratio_p90']:>7.3f} "
                  f"{r['final_ratio_median']:>14.3f} {r['final_ratio_p10']:>7.3f} "
                  f"{r['final_ratio_p90']:>7.3f} {100 * r['final_lt_half']:>8.1f}%")
        for r in ae:
            ms, fs = r["mosaic_shift"], r["final_shift"]
            tm = max(sum(ms.values()), 1); tf = max(sum(fs.values()), 1)
            print(f"    {r['cls']:>7} mosaic 后: 变小 {100*ms['smaller']/tm:5.1f}% 不变 {100*ms['same']/tm:5.1f}% "
                  f"变大 {100*ms['larger']/tm:5.1f}%   |   final 后: 变小 {100*fs['smaller']/tf:5.1f}% "
                  f"不变 {100*fs['same']/tf:5.1f}% 变大 {100*fs['larger']/tf:5.1f}%")
        res[f"aug_effect_{t}"] = ae

    # ---------- 全 300 epoch 的混合估计（close_mosaic=10 -> 290 ON + 10 OFF）----------
    W_ON, W_OFF = 290.0 / 300.0, 10.0 / 300.0
    if "mosaic_ON" in P and "mosaic_OFF" in P:
        print("\n" + "=" * 100)
        print("全 300 epoch 混合估计（close_mosaic=10 ⇒ 290 epoch mosaic-ON + 10 epoch mosaic-OFF，"
              "均为 args.yaml 实测）")
        print("=" * 100)
        print(f"  {'class':>8} {'n':>6} {'ExposureA(ON)':>14} {'ExposureA(OFF)':>15} "
              f"{'ExposureA(mix)':>15} {'fin|inst(mix)':>14}")
        mix = {}
        for cls in ("small", "medium", "large"):
            uo = [u for u, r in P["mosaic_ON"].items() if size_class(r["native_area"]) == cls]
            uf = [u for u, r in P["mosaic_OFF"].items() if size_class(r["native_area"]) == cls]
            if not uo or not uf:
                continue
            eo = float(np.mean([P["mosaic_ON"][u]["exposureA"] for u in uo]))
            ef = float(np.mean([P["mosaic_OFF"][u]["exposureA"] for u in uf]))
            fo = float(np.nanmean([P["mosaic_ON"][u]["p_fin_given_inst"] for u in uo]))
            ff = float(np.nanmean([P["mosaic_OFF"][u]["p_fin_given_inst"] for u in uf]))
            mix[cls] = dict(n=len(uo), exposureA_on=eo, exposureA_off=ef,
                            exposureA_mix=W_ON * eo + W_OFF * ef,
                            p_fin_given_inst_mix=W_ON * fo + W_OFF * ff)
            print(f"  {cls:>8} {len(uo):>6} {eo:>14.4f} {ef:>15.4f} {mix[cls]['exposureA_mix']:>15.4f} "
                  f"{mix[cls]['p_fin_given_inst_mix']:>14.4f}")
        if "small" in mix and "medium" in mix and "large" in mix:
            print(f"  ratio (mix) small/medium = "
                  f"{mix['small']['exposureA_mix'] / max(mix['medium']['exposureA_mix'], 1e-9):.4f} ; "
                  f"small/large = {mix['small']['exposureA_mix'] / max(mix['large']['exposureA_mix'], 1e-9):.4f}")
        res["mixture"] = mix

    (OUT / "_sge_tables.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {OUT / '_sge_tables.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
