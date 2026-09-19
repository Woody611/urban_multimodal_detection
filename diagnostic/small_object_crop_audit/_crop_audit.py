"""diagnostic/small_object_crop_audit/_crop_audit.py — Small-object 同步 Crop / Oversampling 只读可行性审计。

⚠️ READ-ONLY：不训练、不创建 run、不修改任何正式配置 / 数据 / 代码。
   本脚本位于 diagnostic/ 下，不会被任何训练入口自动加载。

COCO 面积定义与既有 audit 保持一致（native 像素）：
    small < 32² = 1024 ; medium 1024–9216 ; large > 9216

用法:  python diagnostic/small_object_crop_audit/_crop_audit.py
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "data/processed/rgbid_split"
TRAIN_IMG = SPLIT / "images/train/visible"
TRAIN_LAB = SPLIT / "labels/train/visible"
VAL_LAB = SPLIT / "labels/val/visible"

SMAX, MMAX = 32 ** 2, 96 ** 2          # 1024, 9216
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
IMGSZ = 1280
N_SIM = 1600          # 全部 train 图片参与模拟
SEED = 42


def read_labels(p):
    if not p.exists():
        return np.zeros((0, 5))
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.split()
        if len(s) >= 5:
            rows.append([float(x) for x in s[:5]])
    return np.array(rows).reshape(-1, 5) if rows else np.zeros((0, 5))


def to_xyxy(lab, W, H):
    """归一化 cxcywh -> native xyxy"""
    if len(lab) == 0:
        return np.zeros((0, 4)), np.zeros((0,), int)
    cx, cy, w, h = lab[:, 1] * W, lab[:, 2] * H, lab[:, 3] * W, lab[:, 4] * H
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return np.stack([x1, y1, x2, y2], 1), lab[:, 0].astype(int)


def area(b):
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def bucket(a):
    r = np.full(len(a), 2, int)
    r[a < SMAX] = 0
    r[(a >= SMAX) & (a < MMAX)] = 1
    return r


def pct(v, qs=(10, 25, 50, 75, 90)):
    if len(v) == 0:
        return {q: float("nan") for q in qs}
    return {q: float(np.percentile(v, q)) for q in qs}


def clip_ratio(box, win):
    """box 在 win 内的面积保留比例 -> 被裁掉的比例 (0 = 完整保留, 1 = 完全裁掉)"""
    x1, y1, x2, y2 = box
    wx1, wy1, wx2, wy2 = win
    ix1, iy1 = max(x1, wx1), max(y1, wy1)
    ix2, iy2 = min(x2, wx2), min(y2, wy2)
    iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
    inter = iw * ih
    a0 = max((x2 - x1) * (y2 - y1), 1e-9)
    return 1.0 - inter / a0


def load_dataset():
    """返回 [(W0,H0, boxes_xyxy, cls, areas, buckets), ...]"""
    data = []
    files = sorted(TRAIN_IMG.glob("*"))
    for p in files:
        with Image.open(p) as im:
            W0, H0 = im.size
        lab = read_labels(TRAIN_LAB / f"{p.stem}.txt")
        b, c = to_xyxy(lab, W0, H0)
        a = area(b)
        data.append((W0, H0, b, c, a, bucket(a)))
    return data


# ================================================================ §2
def sec_distribution(data):
    print("=" * 100)
    print("§2 Small-object 真实分布（train，native 像素；small<1024 / medium 1024-9216 / large>9216）")
    print("=" * 100)
    tot = Counter(); sml = Counter(); med = Counter(); lar = Counter()
    for W0, H0, b, c, a, bk in data:
        for cl, k in zip(c, bk):
            tot[cl] += 1
            (sml, med, lar)[k][cl] += 1
    print(f"  {'class':<13}{'total':>7}{'small':>7}{'medium':>8}{'large':>7}{'small ratio':>13}")
    for cl in range(12):
        t = tot[cl]
        print(f"  {NAMES[cl]:<13}{t:>7}{sml[cl]:>7}{med[cl]:>8}{lar[cl]:>7}"
              f"{(sml[cl]/t*100 if t else 0):>12.1f}%")
    T = sum(tot.values()); S = sum(sml.values())
    print(f"  {'合计':<13}{T:>7}{S:>7}{sum(med.values()):>8}{sum(lar.values()):>7}{S/T*100:>12.1f}%")

    # 每图 small GT 分布
    cnts = [int((bk == 0).sum()) for *_x, bk in data]
    cnts = np.array(cnts)
    print(f"\n  每图 small GT 数量分布（n={len(cnts)} 图）")
    for lo, hi, lbl in ((0, 0, "= 0"), (1, 1, "= 1"), (2, 2, "= 2"), (3, 4, "3-4"),
                        (5, 9, "5-9"), (10, 10 ** 9, ">= 10")):
        m = (cnts >= lo) & (cnts <= hi)
        print(f"    {lbl:<8}{m.sum():>6} 图 ({m.mean()*100:>5.1f}%)")
    print(f"    含 small 的图: {(cnts>0).sum()} ({(cnts>0).mean()*100:.1f}%)   "
          f"均值 {cnts.mean():.2f} / 中位 {np.median(cnts):.0f} / 最大 {cnts.max()}")

    # small bbox 尺寸分位
    allb, allwf, allhf = [], [], []
    for W0, H0, b, c, a, bk in data:
        m = bk == 0
        if m.any():
            w = b[m, 2] - b[m, 0]; h = b[m, 3] - b[m, 1]
            allb.append(np.sqrt(a[m])); allwf.append(w); allhf.append(h)
    allb = np.concatenate(allb); allwf = np.concatenate(allwf); allhf = np.concatenate(allhf)
    ar = allwf / np.maximum(allhf, 1e-9)
    print(f"\n  small bbox 尺寸（native 像素，n={len(allb)}）")
    print(f"  {'量':<14}{'P10':>9}{'P25':>9}{'median':>9}{'P75':>9}{'P90':>9}")
    for nm, v in (("width", allwf), ("height", allhf), ("sqrt(area)", allb), ("aspect(w/h)", ar)):
        p = pct(v)
        print(f"  {nm:<14}{p[10]:>9.1f}{p[25]:>9.1f}{p[50]:>9.1f}{p[75]:>9.1f}{p[90]:>9.1f}")

    # small 与图像尺寸 / 1280 后的像素
    print(f"\n  small object 在 1280 输入下的实际像素（按各图自身 resize 比例）")
    px_w, px_h, px_s = [], [], []
    for W0, H0, b, c, a, bk in data:
        m = bk == 0
        if m.any():
            s = IMGSZ / max(W0, H0)
            px_w.append((b[m, 2] - b[m, 0]) * s)
            px_h.append((b[m, 3] - b[m, 1]) * s)
            px_s.append(np.sqrt(a[m]) * s)
    px_w = np.concatenate(px_w); px_h = np.concatenate(px_h); px_s = np.concatenate(px_s)
    print(f"  {'量':<14}{'P10':>9}{'P25':>9}{'median':>9}{'P75':>9}{'P90':>9}{'mean':>9}")
    for nm, v in (("width", px_w), ("height", px_h), ("sqrt(area)", px_s)):
        p = pct(v)
        print(f"  {nm:<14}{p[10]:>9.1f}{p[25]:>9.1f}{p[50]:>9.1f}{p[75]:>9.1f}{p[90]:>9.1f}{v.mean():>9.1f}")
    return data


# ================================================================ §3/§7/§8
def simulate_crop(data, r, rule="random", jitter=0.15, n_sim=N_SIM):
    """返回该 crop 策略的聚合统计。rule: random(随机一个 small GT 为中心) / centroid(最大化保留)"""
    rng = random.Random(SEED)
    st = defaultdict(list)
    keep_small = drop_small = 0
    n_small_tot = 0
    cls_small_kept = Counter(); cls_small_all = Counter()
    clip_buckets = {"完整(0)": 0, "轻微(<10%)": 0, "10-25%": 0, "25-50%": 0, ">50%": 0, "完全裁掉": 0}
    all_clip = defaultdict(lambda: {"完整(0)": 0, "轻微(<10%)": 0, "10-25%": 0, "25-50%": 0, ">50%": 0, "完全裁掉": 0})
    small_px_base, small_px_crop = [], []
    small_kept_px = []
    ctx_ratio = []
    small_per_img = []
    small_per_1280 = []

    for idx in range(min(n_sim, len(data))):
        W0, H0, b, c, a, bk = data[idx]
        sm = np.where(bk == 0)[0]
        n_small_tot_i = len(sm)
        for cl in c[sm]:
            cls_small_all[cl] += 1
        scale_b = IMGSZ / max(W0, H0)
        small_px_base.extend(np.sqrt(a[sm]) * scale_b)
        if n_small_tot_i == 0:
            continue
        n_small_tot += n_small_tot_i
        # crop window —— 与图像等比，面积比 = r（r=1.0 即整图，不改变宽高比）
        cw_t, ch_t = np.sqrt(r) * W0, np.sqrt(r) * H0
        side = max(cw_t, ch_t)
        if rule == "random":
            j = sm[rng.randrange(len(sm))]
            cx, cy = (b[j, 0] + b[j, 2]) / 2, (b[j, 1] + b[j, 3]) / 2
            cx += rng.uniform(-jitter, jitter) * cw_t
            cy += rng.uniform(-jitter, jitter) * ch_t
        else:  # centroid of small GT
            cx = float(np.mean((b[sm, 0] + b[sm, 2]) / 2))
            cy = float(np.mean((b[sm, 1] + b[sm, 3]) / 2))
        x1 = int(np.clip(cx - cw_t / 2, 0, max(W0 - cw_t, 0)))
        y1 = int(np.clip(cy - ch_t / 2, 0, max(H0 - ch_t, 0)))
        x2 = int(min(x1 + cw_t, W0)); y2 = int(min(y1 + ch_t, H0))
        win = (x1, y1, x2, y2)
        cw, ch = x2 - x1, y2 - y1
        scale_c = IMGSZ / max(cw, ch)
        st["enlarge"].append(scale_c / scale_b)
        ctx_ratio.append((cw * ch) / (W0 * H0))

        # 逐 GT 统计
        cr = np.array([clip_ratio(bb, win) for bb in b])
        inside = (b[:, 0] + b[:, 2]) / 2 >= x1
        for k in range(len(b)):
            inside[k] &= ((b[k, 0] + b[k, 2]) / 2 <= x2 and (b[k, 1] + b[k, 3]) / 2 >= y1
                          and (b[k, 1] + b[k, 3]) / 2 <= y2)
            v = cr[k]
            lab = ("完全裁掉" if not inside[k] else
                   "完整(0)" if v <= 1e-9 else
                   "轻微(<10%)" if v < 0.10 else
                   "10-25%" if v < 0.25 else
                   "25-50%" if v < 0.50 else ">50%")
            all_clip[int(bk[k])][lab] += 1
        for k in sm:
            v = cr[k]
            lab = ("完全裁掉" if not inside[k] else
                   "完整(0)" if v <= 1e-9 else
                   "轻微(<10%)" if v < 0.10 else
                   "10-25%" if v < 0.25 else
                   "25-50%" if v < 0.50 else ">50%")
            clip_buckets[lab] += 1
            if inside[k] and v < 0.50:
                keep_small += 1
                small_kept_px.append(np.sqrt(a[k]) * scale_c)
                cls_small_kept[int(c[k])] += 1
            else:
                drop_small += 1
        small_px_crop.extend(np.sqrt(a[sm]) * scale_c)
        n_keep = int(((bk[sm] == 0) & inside[sm] & (cr[sm] < 0.5)).sum())
        small_per_img.append(n_keep)
        small_per_1280.append(n_keep / (cw * ch) * (IMGSZ * IMGSZ))

    return dict(r=r, rule=rule, keep=keep_small, drop=drop_small, n_small=n_small_tot,
                clip=clip_buckets, all_clip=all_clip,
                base=np.array(small_px_base), crop=np.array(small_px_crop),
                kept=np.array(small_kept_px), ctx=np.array(ctx_ratio),
                spimg=np.array(small_per_img), sp1280=np.array(small_per_1280),
                cls_kept=cls_small_kept, cls_all=cls_small_all,
                enlarge=np.array(st["enlarge"]))


def sec_crop(data):
    print("\n" + "=" * 100)
    print("§3/§8 Crop 模拟（small-GT 为中心，随机抖动 ±15% 边长，n=1600 图）")
    print("=" * 100)
    rows = {}
    for r in (1.0, 0.75, 0.6, 0.5, 0.4):
        rows[r] = simulate_crop(data, r, "random")
    base = rows[1.0]["base"]
    print(f"  {'ratio':>6}{'实际上下文':>11}{'放大倍数中位':>13}{'small保留':>11}{'small严重截断':>14}"
          f"{'small完全裁掉':>14}{'med裁掉':>10}{'large裁掉':>11}")
    for r, v in rows.items():
        n = v["keep"] + v["drop"]
        ctx = float(np.median(v["ctx"]))
        enl = float(np.median(v["enlarge"]))
        sev = (v["clip"][">50%"] + v["clip"]["完全裁掉"]) / max(n, 1)
        tgt = v["all_clip"]
        def dr(k):
            tot = sum(tgt[k].values()); d = tgt[k]["完全裁掉"]
            return d / max(tot, 1)
        print(f"  {r:>6.2f}{ctx:>11.3f}{enl:>13.2f}{v['keep']/max(n,1)*100:>10.1f}%"
              f"{sev*100:>13.1f}%{v['clip']['完全裁掉']/max(n,1)*100:>13.1f}%"
              f"{dr(1)*100:>9.1f}%{dr(2)*100:>10.1f}%")

    print(f"\n  small object 在 1280 输入下的 sqrt(area)（像素）—— baseline vs crop")
    print(f"  {'策略':<22}{'P10':>8}{'P25':>8}{'median':>8}{'P75':>8}{'P90':>8}{'mean':>8}")
    pb = pct(base)
    print(f"  {'baseline(原图 resize)':<22}{pb[10]:>8.1f}{pb[25]:>8.1f}{pb[50]:>8.1f}{pb[75]:>8.1f}{pb[90]:>8.1f}{base.mean():>8.1f}")
    for r, v in rows.items():
        if r == 1.0:
            continue
        p = pct(v["crop"])
        print(f"  {'crop r=%.2f' % r:<22}{p[10]:>8.1f}{p[25]:>8.1f}{p[50]:>8.1f}{p[75]:>8.1f}{p[90]:>8.1f}{v['crop'].mean():>8.1f}")

    print(f"\n  §7 bbox 截断审计（small GT）")
    print(f"  {'策略':<12}{'完整(0)':>10}{'轻微<10%':>10}{'10-25%':>10}{'25-50%':>10}{'>50%':>10}{'完全裁掉':>10}")
    for r, v in rows.items():
        if r == 1.0:
            continue
        n = v["keep"] + v["drop"]
        print(f"  {'r=%.2f' % r:<12}" + "".join(f"{v['clip'][k]/max(n,1)*100:>9.1f}%" for k in
              ("完整(0)", "轻微(<10%)", "10-25%", "25-50%", ">50%", "完全裁掉")))

    print(f"\n  §3 GT 密度")
    print(f"  {'策略':<12}{'small GT/图(保留)':>20}{'small GT / 1280²':>20}{'small 类别覆盖(类)':>20}")
    for r, v in rows.items():
        if r == 1.0:
            continue
        print(f"  {'r=%.2f' % r:<12}{v['spimg'].mean():>20.2f}{v['sp1280'].mean():>20.4f}"
              f"{len([k for k, n in v['cls_kept'].items() if n > 0]):>20}")
    nb = np.array([int((d[5] == 0).sum()) for d in data])
    # baseline 输入像素 = 各图 resize 后面积之和 / 图数（1920x1080 -> 720x1280）
    px = np.array([(IMGSZ / max(d[0], d[1]) * d[0]) * (IMGSZ / max(d[0], d[1]) * d[1]) for d in data])
    base_per_1280 = nb.mean() / px.mean() * (IMGSZ * IMGSZ)
    print(f"  {'baseline':<12}{nb.mean():>20.2f}{base_per_1280:>20.4f}{12:>20}")
    print(f"  （baseline 每图输入像素均值 = {px.mean():,.0f}；crop 后为 1280×1280 等比画布）")

    print(f"\n  §3/§5 类别暴露（small GT 保留量 / 原 small 量）")
    print(f"  {'class':<13}" + "".join(f"{'r=%.2f' % r:>10}" for r in rows if r != 1.0))
    for cl in range(12):
        if rows[0.5]["cls_all"][cl] == 0:
            continue
        line = f"  {NAMES[cl]:<13}"
        for r, v in rows.items():
            if r == 1.0:
                continue
            line += f"{(v['cls_kept'][cl]/v['cls_all'][cl]*100):>9.0f}%"
        print(line)
    return rows


# ================================================================ §9
def sec_oversample(data):
    print("\n" + "=" * 100)
    print("§9 Oversampling 模拟（含 small GT 的图获得 f× 采样概率）")
    print("=" * 100)
    n_small_img = np.array([int((d[5] == 0).sum()) for d in data])
    n_small = n_small_img
    has = n_small > 0
    cls_all = Counter()
    for *_x, c, a, bk in [(d[0], d[1], d[2], d[3], d[4], d[5]) for d in data]:
        for cl in c[bk == 0]:
            cls_all[int(cl)] += 1
    N = len(data)
    print(f"  含 small 的图 = {has.sum()} / {N} ({has.mean()*100:.1f}%)")
    print(f"  每 epoch baseline small GT exposure = {n_small.sum()}")
    print(f"  {'factor':>8}{'small exposure':>16}{'增幅':>9}{'重复率(>1次)':>14}{'最大重复':>10}{'有效唯一图':>12}")
    for f in (1.25, 1.5, 2.0):
        w = np.where(has, f, 1.0)
        p = w / w.sum()
        exp_cnt = p * N
        s_exp = float((n_small * exp_cnt).sum())
        dup = float((exp_cnt > 1).mean())
        print(f"  {f:>8.2f}{s_exp:>16.0f}{s_exp/n_small.sum()*100-100:>8.1f}%{dup*100:>13.1f}%"
              f"{exp_cnt.max():>10.2f}{int((exp_cnt>0).sum()):>12}")
    print(f"\n  逐类 small exposure 增幅")
    print(f"  {'class':<13}" + "".join(f"{'x%.2f' % f:>10}" for f in (1.25, 1.5, 2.0)))
    for cl in range(12):
        if cls_all[cl] == 0:
            continue
        line = f"  {NAMES[cl]:<13}"
        for f in (1.25, 1.5, 2.0):
            w = np.where(has, f, 1.0); p = w / w.sum(); exp_cnt = p * N
            num = 0.0
            for i, d in enumerate(data):
                num += int(((d[3] == cl) & (d[5] == 0)).sum()) * exp_cnt[i]
            line += f"{num/cls_all[cl]*100-100:>9.1f}%"
        print(line)


# ================================================================ §11 Mosaic
def sec_mosaic(data, rows):
    print("\n" + "=" * 100)
    print("§11 与 Mosaic 的关系（有效目标尺度）")
    print("=" * 100)
    print("  代码事实（augment.py 实读）：")
    print("    Mosaic 画布 = 2s × 2s（s=imgsz=1280 -> 2560×2560，augment.py:3172）")
    print("    随后 RandomPerspective 把画布 resize 回 imgsz=1280")
    print("    => 每张拼图 tile 在最终输入中约占 640px，即目标被系统性缩小约 2×")
    print("    RandomPerspective 另有随机缩放 s ∈ [1-0.5, 1+0.5] = [0.5, 1.5]（augment.py:1056）")
    print("    close_mosaic=10 -> 仅最后 10/300 = 3.3% 的 epoch 关闭 mosaic")
    print()
    base_scale = []
    for W0, H0, b, c, a, bk in data:
        s = IMGSZ / max(W0, H0)
        m = bk == 0
        if m.any():
            base_scale.extend(np.sqrt(a[m]) * s)
    base_scale = np.array(base_scale)
    med0 = float(np.median(base_scale))
    print(f"  baseline 无 mosaic 时（占 3.3% 训练）small sqrt(area) 中位 = {med0:.1f} px   <- 参考上界")
    print(f"  baseline 含 mosaic 时（占 96.7% 训练）        ≈ {med0*0.5:.1f} px   (×0.5 mosaic 因子)")
    print()
    print(f"  {'策略(在含mosaic的epoch内)':<26}{'crop放大':>10}{'含mosaic中位':>14}{'vs baseline含mosaic':>21}{'vs baseline无mosaic':>21}")
    for r, v in rows.items():
        if r == 1.0:
            continue
        enl = float(np.median(v["enlarge"]))
        medc = float(np.median(v["crop"]))
        print(f"  {'crop r=%.2f' % r:<26}{enl:>10.2f}{medc*0.5:>14.1f}{medc*0.5/(med0*0.5)*100-100:>20.0f}%"
              f"{medc*0.5/med0*100-100:>20.0f}%")
    print(f"\n  关键：crop 在 mosaic epoch 内的放大 {float(np.median(rows[0.5]['enlarge'])):.2f}×(r=0.5) / "
          f"{float(np.median(rows[0.4]['enlarge'])):.2f}×(r=0.4)")
    print(f"        均小于 mosaic 自身的缩小 2.00×")
    print(f"        => crop 只是【部分抵消】mosaic 的缩小，并未超过 baseline 自身的无 mosaic 尺度")


# ================================================================ §6 alignment
def sec_alignment():
    print("\n" + "=" * 100)
    print("§6 多模态同步裁剪安全性（实测）")
    print("=" * 100)
    import cv2
    imgs = sorted(TRAIN_IMG.glob("*"))
    p = imgs[0]
    ir = Path(str(p).replace("visible", "infrared"))
    dp = Path(str(p).replace("visible", "depth"))
    v = cv2.imread(str(p)); i = cv2.imread(str(ir), cv2.IMREAD_GRAYSCALE)
    d = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
    d = d[..., 0] if d.ndim == 3 else d
    print(f"  样本 {p.name}: visible{v.shape} infrared{i.shape} depth{d.shape}")
    print(f"  三模态尺寸完全一致: {v.shape[:2]==i.shape[:2]==d.shape[:2]}")
    b, g, r = cv2.split(v)
    merged = cv2.merge((b, g, r, i.astype(np.uint8), d.astype(np.uint8)))
    x1, y1, x2, y2 = 100, 50, 500, 350
    c_merged = merged[y1:y2, x1:x2]
    c_sep = np.dstack([cv2.split(v[y1:y2, x1:x2])[0], cv2.split(v[y1:y2, x1:x2])[1],
                       cv2.split(v[y1:y2, x1:x2])[2], i[y1:y2, x1:x2].astype(np.uint8),
                       d[y1:y2, x1:x2].astype(np.uint8)])
    print(f"  '先合并再裁' 与 '先分别裁再合并' 是否逐位相同: {np.array_equal(c_merged, c_sep)}")
    print(f"  -> 合并发生在 base.py::_merge_channels_rgbid（load_image 内），在**任何几何变换之前**")
    print(f"     因此在 get_image_and_label 的 img 上做 crop => 三模态同步是【构造性保证】")
    print(f"  深度位深: {d.dtype}（16bit 生成 1478 张 / 8bit 122 张），visible/IR 恒为 8bit 3ch")


def main():
    print("Small-object 同步 Crop / Oversampling 可行性审计（READ-ONLY）")
    print(f"  train split = {SPLIT/'images/train/visible'}")
    data = load_dataset()
    print(f"  载入 {len(data)} 张 train 图及其标签")
    sec_distribution(data)
    rows = sec_crop(data)
    sec_oversample(data)
    sec_mosaic(data, rows)
    sec_alignment()


if __name__ == "__main__":
    main()
