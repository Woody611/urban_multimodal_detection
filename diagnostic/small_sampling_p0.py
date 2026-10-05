"""diagnostic/small_sampling_p0.py — Small-object Sampling P0 (ZERO GPU, read-only + CPU sim).

Answers, with simulation rather than assertion:
  * is there a safe image-level small-object sampling insertion point?
  * does it actually raise small-object *training exposure* (not just duplicate count)?
  * does it break epoch / optimizer-step / LR-schedule / Mosaic / reproducibility?
  * does it create class bias?
  * is it worth a GPU run?

Reuses the project's canonical small-object definition (do NOT redefine):
    diagnostic/gt_level_attribution.py::SIZE_BINS  ->  small = native area < 1024 px^2

Real pipeline facts this script is built on (read from source, not assumed):
    ultralytics/data/build.py::InfiniteDataLoader / _RepeatSampler / build_dataloader
    ultralytics/data/augment.py::Mosaic.get_indexes   -> random.choices(dataset.buffer, k=3)
    ultralytics/data/base.py:676-683                  -> buffer = strict FIFO of last
                                                         min(ni, batch*8, 1000) LOADED indices
"""

from __future__ import annotations

import json
import os
import random
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]

# ---- canonical, reused (do not redefine) ----
SMALL_MAX_AREA = 1024.0          # gt_level_attribution.SIZE_BINS: small in [0, 1024)
MEDIUM_MAX_AREA = 9216.0

# ---- training geometry (from configs/train_rgbid_sepstem_clahe.yaml + default.yaml) ----
N_TRAIN, VAL_N = 1600, 400
BATCH = 8
EPOCHS = 300
CLOSE_MOSAIC = 10
MOSAIC_P = 1.0
IMGSZ = 1280
BUFFER_MAX = min(N_TRAIN, BATCH * 8, 1000)      # = 64, straight from base.py:329
CLASSES = ["person", "bicycle", "car", "boat", "uav", "light", "garbage_can",
           "tricycle", "animal", "sign", "seat", "ball"]
SEED = 42


def img_size(p: Path) -> tuple[int, int]:
    with open(p, "rb") as f:
        head = f.read(32)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", head[16:24])
            return int(w), int(h)
        # JPEG: walk the marker segments to the first SOFn
        f.seek(2)
        while True:
            b = f.read(1)
            while b and b != b"\xff":
                b = f.read(1)
            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)
            if not marker:
                raise ValueError(f"bad jpeg {p}")
            m = marker[0]
            if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                ln = struct.unpack(">H", f.read(2))[0]
                prec, h, w = struct.unpack(">BHH", f.read(5))[:3]
                return int(w), int(h)
            if m in (0xD8, 0xD9) or 0xD0 <= m <= 0xD7:
                continue
            ln = struct.unpack(">H", f.read(2))[0]
            f.seek(ln - 2, 1)


def load_split(img_dir: Path, lbl_dir: Path):
    """Per-image: native (w,h), per-class instance areas in native px^2."""
    out = []
    for img in sorted(img_dir.iterdir()):
        lbl = lbl_dir / (img.stem + ".txt")
        w, h = img_size(img)
        cls, areas = [], []
        if lbl.exists():
            for ln in lbl.read_text(encoding="utf-8", errors="ignore").splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                cls.append(int(float(p[0])))
                areas.append(float(p[3]) * w * float(p[4]) * h)   # native px^2
        out.append(dict(stem=img.stem, w=w, h=h, cls=np.array(cls, dtype=int),
                        area=np.array(areas, dtype=float)))
    return out


# ============================================================
# Exposure model
# ============================================================
def run_sampler(kind: str, w_small: float, n_images: int, rng: random.Random):
    """Yield image indices in dataloader-consumption order for ONE epoch."""
    if kind == "baseline":                       # RandomSampler, without replacement
        idx = list(range(n_images))
        rng.shuffle(idx)
        return idx
    # weighted, WITH replacement, same length -> optimizer budget preserved
    weights = [w_small if IS_SMALL_IMAGE[i] else 1.0 for i in range(n_images)]
    return rng.choices(range(n_images), weights=weights, k=n_images)


def simulate_epochs(make_order, small_gt, small_img, n_epochs, rng, mosaic_p=MOSAIC_P):
    """Faithful replay of base.py's FIFO buffer + Mosaic.get_indexes(buffer=True).

    Per training sample the dataset loads, in order: the centre index, then the 3
    side indices chosen by `random.choices(buffer, k=3)` -- and every one of those
    four loads is appended to the buffer (base.py:679).
    """
    buf: list[int] = []
    st = dict(samples=0, centre_small_img=0, slot_small_img=0, slots=0,
              centre_small_gt=0.0, mosaic_small_gt=0.0, mosaic_total_gt=0.0,
              cls_small=Counter(), cls_total=Counter(), seen=Counter(),
              buf_small=0, buf_len=0, ep_unique=[])
    for ep in range(n_epochs):
        order = make_order(ep)
        ep_seen = Counter()
        for centre in order:
            ep_seen[centre] += 1
            st["samples"] += 1
            st["seen"][centre] += 1
            buf.append(centre)
            if 1 < len(buf) >= BUFFER_MAX:
                buf.pop(0)

            if rng.random() < mosaic_p and len(buf) > 0:
                # small-image share of the pool Mosaic actually draws from
                st["buf_small"] += sum(1 for j in buf if small_img[j])
                st["buf_len"] += len(buf)
                sides = rng.choices(buf, k=3)
            else:
                sides = []
            for s in sides:
                buf.append(s)
                if 1 < len(buf) >= BUFFER_MAX:
                    buf.pop(0)

            slots = [centre] + sides
            st["slots"] += len(slots)
            for j in slots:
                st["seen"][j] += 1
                if small_img[j]:
                    st["slot_small_img"] += 1
                st["mosaic_small_gt"] += small_gt[j]
                st["mosaic_total_gt"] += len(CLS[j])
                for c in CLS[j][SMALL[j]]:
                    st["cls_small"][int(c)] += 1
                for c in CLS[j]:
                    st["cls_total"][int(c)] += 1
            if small_img[centre]:
                st["centre_small_img"] += 1
            st["centre_small_gt"] += small_gt[centre]
        st["ep_unique"].append(len(ep_seen))
    return st, buf


def main():
    out_dir = ROOT / "diagnostic" / "small_sampling_p0"
    out_dir.mkdir(parents=True, exist_ok=True)
    R: dict = {}

    tr_img = ROOT / "data/processed/rgbid_split_train/images/train/visible"
    tr_lbl = ROOT / "data/processed/rgbid_split_train/labels/train/visible"
    va_img = ROOT / "data/processed/rgbid_split_train/images/val/visible"
    va_lbl = ROOT / "data/processed/rgbid_split_train/labels/val/visible"

    print("loading train labels ...")
    train = load_split(tr_img, tr_lbl)
    val = load_split(va_img, va_lbl)

    global IS_SMALL_IMAGE, CLS, SMALL
    n = len(train)
    areas = [t["area"] for t in train]
    CLS = [t["cls"] for t in train]
    SMALL = [t["area"] < SMALL_MAX_AREA for t in train]
    small_gt = [int(s.sum()) for s in SMALL]
    IS_SMALL_IMAGE = [bool(s) for s in small_gt]
    is_small_img = np.array(IS_SMALL_IMAGE)

    print("=" * 100)
    print("§3  BASELINE CENSUS  (canonical def: native area < 1024 px^2)")
    print("=" * 100)
    tot_gt = int(sum(len(a) for a in areas))
    tot_small = int(sum(small_gt))
    n_small_img = int(is_small_img.sum())
    census = dict(
        n_train=n, val_n=len(val),
        total_gt=tot_gt, small_gt=tot_small,
        small_ratio=tot_small / tot_gt,
        images_with_small=n_small_img,
        mean_small_per_image=tot_small / n,
        median_small_per_image=float(np.median(small_gt)),
        max_small_per_image=int(max(small_gt)),
        small_image_fraction=n_small_img / n,
        n_classes_present=len(set(int(c) for t in train for c in t["cls"])),
    )
    for k, v in census.items():
        print(f"  {k:<26} {v}")
    R["census"] = census

    # native vs model-space (1280 letterbox) size
    def scale2(t):
        return min(IMGSZ / t["w"], IMGSZ / t["h"]) ** 2

    nat_small_inst = np.concatenate([a[a < SMALL_MAX_AREA] for a in areas if len(a)]) if tot_gt else np.array([])
    m1280_small_inst = np.concatenate([
        a * scale2(t) for t, a in zip(train, areas) if len(a) and (a * scale2(t) < SMALL_MAX_AREA).any()
    ]) if tot_gt else np.array([])
    m1280_small_inst = m1280_small_inst[m1280_small_inst < SMALL_MAX_AREA] if len(m1280_small_inst) else m1280_small_inst
    n_small_at1280 = int(sum(int((a * scale2(t) < SMALL_MAX_AREA).sum()) for t, a in zip(train, areas)))
    print(f"\n  native small sqrt(area) p50 = {np.sqrt(np.median(nat_small_inst)):.2f} px  (n={len(nat_small_inst)})")
    if len(m1280_small_inst):
        print(f"  those instances @1280 sqrt(area) p50 = {np.sqrt(np.median(m1280_small_inst)):.2f} px")
    print(f"  instances small by NATIVE area = {tot_small}   by @1280 area = {n_small_at1280}   "
          f"(Δ = {n_small_at1280 - tot_small:+d})")
    R["size_px"] = dict(
        native_sqrt_p50=float(np.sqrt(np.median(nat_small_inst))) if len(nat_small_inst) else None,
        n_small_native=tot_small, n_small_at1280=n_small_at1280,
    )

    # per-class small composition
    print("\n  small-GT composition by class:")
    cls_small_tot = Counter()
    cls_all_tot = Counter()
    for t in train:
        for c in t["cls"]:
            cls_all_tot[int(c)] += 1
        for c in t["cls"][t["area"] < SMALL_MAX_AREA]:
            cls_small_tot[int(c)] += 1
    rows = []
    print(f"    {'class':<13}{'small':>7}{'all':>7}{'small%':>9}")
    for c in range(12):
        s, a = cls_small_tot[c], cls_all_tot[c]
        rows.append(dict(cls=c, name=CLASSES[c], small=s, all=a, frac=(s / a if a else 0.0)))
        print(f"    {CLASSES[c]:<13}{s:>7}{a:>7}{100*(s/a if a else 0):>8.1f}%")
    R["class_small"] = rows

    # ============================================================
    print("\n" + "=" * 100)
    print("§8/§9  OPTIMIZER-STEP INTEGRITY")
    print("=" * 100)
    steps_base = -(-n // BATCH)
    print(f"  baseline : len(dataset)={n}  steps/epoch=ceil({n}/{BATCH})={steps_base}  "
          f"total={steps_base*EPOCHS:,}")
    for w in (1.25, 1.5, 1.75, 2.0):
        n_a = int(round(w * n_small_img + (n - n_small_img)))
        steps_a = -(-n_a // BATCH)
        print(f"  Design A w={w:<5} len={n_a}  steps/epoch={steps_a}  "
              f"total={steps_a*EPOCHS:,}  Δ={100*(steps_a/steps_base-1):+.1f}%  ← CONFOUND")
    print(f"  Design B w=*     len={n}  steps/epoch={steps_base}  "
          f"total={steps_base*EPOCHS:,}  Δ=+0.0%   ← PRESERVED")
    R["optimizer"] = dict(steps_per_epoch=steps_base, total=steps_base * EPOCHS,
                          designA={str(w): -(-int(round(w * n_small_img + (n - n_small_img))) // BATCH) for w in (1.25, 1.5, 1.75, 2.0)})

    # ============================================================
    print("\n" + "=" * 100)
    print("§10/§14  EXPOSURE SIMULATION  (design B: weighted, length preserved)")
    print("   exposure is computed through the REAL Mosaic buffer mechanism,")
    print("   not by assuming 'sampling gain == model-visible gain'.")
    print("=" * 100)
    N_EP = 20
    results = {}
    hdr = (f"  {'config':<12}{'centre small%':>14}{'slot small%':>13}{'smallGT/sample':>15}"
           f"{'vs base':>9}{'dup%':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    base_stats = None
    for label, kind, w in ([("baseline", "baseline", 1.0)] +
                           [(f"w={w}", "weighted", w) for w in (1.25, 1.5, 1.75, 2.0)]):
        rng = random.Random(SEED)
        seq = (lambda ep, k=kind, ww=w: run_sampler(k, ww, n, rng))
        st, _ = simulate_epochs(seq, small_gt, IS_SMALL_IMAGE, N_EP, rng)
        centre_p = st["centre_small_img"] / st["samples"]
        slot_p = st["slot_small_img"] / st["slots"]
        gt_per_sample = st["mosaic_small_gt"] / st["samples"]
        if base_stats is None:
            base_stats = st
            gain = 1.0
        else:
            gain = gt_per_sample / (base_stats["mosaic_small_gt"] / base_stats["samples"])
        dup = 1 - len(st["seen"]) / st["slots"]
        results[label] = dict(centre_small_frac=centre_p, slot_small_frac=slot_p,
                              small_gt_per_sample=gt_per_sample, gain=gain, dup_frac=dup,
                              buf_small_frac=(st["buf_small"] / max(st["buf_len"], 1)),
                              cls_small={int(k): v for k, v in st["cls_small"].items()},
                              cls_total={int(k): v for k, v in st["cls_total"].items()},
                              samples=st["samples"])
        print(f"  {label:<12}{100*centre_p:>13.2f}%{100*slot_p:>12.2f}%{gt_per_sample:>15.3f}"
              f"{gain:>9.3f}{100*dup:>7.1f}%")
    R["exposure"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("cls_")}
                     for k, v in results.items()}

    print("\n  ⚠ 'centre small%' = sampler-side effect; 'slot small%' = what Mosaic actually composes.")
    print("  ⚠ Mosaic composes 4 slots/sample, so slot exposure is the model-visible one.")

    # ============================================================
    print("\n" + "=" * 100)
    print("§11  CLASS BIAS  (small-GT exposure change per class)")
    print("=" * 100)
    b = results["baseline"]
    tbl = []
    print(f"  {'class':<13}{'base slots':>12}{'w=1.5 slots':>13}{'Δ%':>9}{'w=2.0 slots':>13}{'Δ%':>9}")
    for c in range(12):
        bs = b["cls_small"].get(c, 0)
        s15 = results["w=1.5"]["cls_small"].get(c, 0)
        s20 = results["w=2.0"]["cls_small"].get(c, 0)
        d15 = 100 * (s15 / bs - 1) if bs else float("nan")
        d20 = 100 * (s20 / bs - 1) if bs else float("nan")
        tbl.append(dict(cls=c, name=CLASSES[c], base=bs, w15=s15, d15=d15, w20=s20, d20=d20))
        print(f"  {CLASSES[c]:<13}{bs:>12}{s15:>13}{d15:>8.1f}%{s20:>13}{d20:>8.1f}%")
    R["class_bias"] = tbl
    g = [t["d20"] for t in tbl if t["base"] > 0 and t["d20"] == t["d20"]]
    print(f"\n  small-GT exposure Δ%(w=2.0): min={min(g):+.1f}%  max={max(g):+.1f}%  "
          f"spread={max(g)-min(g):.1f}pp")
    print(f"  ⇒ all small classes scale together by construction (the rule is image-level,")
    print(f"    not class-level) — but their BASE frequencies differ, so absolute volume differs.")

    # ============================================================
    print("\n" + "=" * 100)
    print("§12  UNIQUE-IMAGE COVERAGE **PER EPOCH**  (the scale that matters)")
    print("=" * 100)
    print(f"  {'config':<12}{'uniq img/ep':>13}{'/1600':>8}{'small uniq':>12}{'non-small uniq':>16}"
          f"{'small slots unused/epoch':>26}")
    for label, kind, w in ([("baseline", "baseline", 1.0)] +
                           [(f"w=1.25", "weighted", 1.25), (f"w=1.5", "weighted", 1.5),
                            (f"w=1.75", "weighted", 1.75), (f"w=2.0", "weighted", 2.0)]):
        rng = random.Random(SEED + 2)
        uu, su, mu = [], [], []
        for _ in range(200):
            seen = Counter()
            for i in run_sampler(kind, w, n, rng):
                seen[i] += 1
            uu.append(len(seen))
            su.append(sum(1 for i in seen if IS_SMALL_IMAGE[i]))
            mu.append(sum(1 for i in seen if not IS_SMALL_IMAGE[i]))
        u, s, m = float(np.mean(uu)), float(np.mean(su)), float(np.mean(mu))
        print(f"  {label:<12}{u:>13.0f}{u/n:>8.3f}{s:>12.0f}{m:>16.0f}"
              f"{100*(1-s/n_small_img):>25.1f}%")
        results.setdefault(label, {})["unique_per_epoch"] = u
        results[label]["unique_small_per_epoch"] = s
        results[label]["unique_nonsmall_per_epoch"] = m
    print("  baseline = RandomSampler without replacement ⇒ every image exactly once/epoch.")
    print("  weighting = with replacement ⇒ high-weight images repeat, low-weight images")
    print("  are missed. ⚠ OVER 100 EPOCHS coverage saturates (100x1600 draws >> 1600),")
    print("  which is why the per-epoch scale is the only informative one.")

    # small-image fraction shift
    print("\n  small-image share of training draws:")
    for label, w in [("baseline", 1.0), ("w=1.25", 1.25), ("w=1.5", 1.5),
                     ("w=1.75", 1.75), ("w=2.0", 2.0)]:
        p = (w * n_small_img) / (w * n_small_img + (n - n_small_img))
        print(f"    {label:<10} {100*p:>6.2f}%   (baseline {100*n_small_img/n:.2f}%, "
              f"over-rep {p/(n_small_img/n):.2f}x)")

    print("\n" + "=" * 100)
    print("§10b  THE AMPLIFICATION MECHANISM (why mosaic gain > draw gain)")
    print("=" * 100)
    print(f"  {'config':<12}{'draw small%':>13}{'BUFFER small%':>15}{'draw gain':>11}"
          f"{'mosaic gain':>13}")
    for label, w in [("baseline", 1.0), ("w=1.25", 1.25), ("w=1.5", 1.5),
                     ("w=1.75", 1.75), ("w=2.0", 2.0)]:
        p_draw = (w * n_small_img) / (w * n_small_img + (n - n_small_img))
        pb = results[label]["buf_small_frac"] if "buf_small_frac" in results.get(label, {}) else None
        print(f"  {label:<12}{100*p_draw:>12.2f}%{(100*pb if pb is not None else float('nan')):>14.2f}%"
              f"{w * n / (w * n_small_img + (n - n_small_img)):>11.3f}{results[label]['gain']:>13.3f}")
    print("  naive draw-level gain = 1600w/(401w+1199); mosaic gain is measured end-to-end.")

    print("\n" + "=" * 100)
    print("§11b  TOTAL (not just small) CLASS EXPOSURE — does the class mix shift?")
    print("=" * 100)
    print(f"  {'class':<13}{'base ALL slots':>15}{'w=2.0 ALL':>12}{'Δ%':>9}"
          f"{'base small':>12}{'w=2.0 small':>12}{'Δ%':>9}")
    for c in range(12):
        ba = b["cls_total"].get(c, 0)
        a20 = results["w=2.0"]["cls_total"].get(c, 0)
        bs = b["cls_small"].get(c, 0)
        s20 = results["w=2.0"]["cls_small"].get(c, 0)
        print(f"  {CLASSES[c]:<13}{ba:>15}{a20:>12}"
              f"{(100*(a20/ba-1) if ba else float('nan')):>8.1f}%"
              f"{bs:>12}{s20:>12}{(100*(s20/bs-1) if bs else float('nan')):>8.1f}%")
    R["total_class_exposure"] = {
        CLASSES[c]: dict(base=b["cls_total"].get(c, 0), w20=results["w=2.0"]["cls_total"].get(c, 0))
        for c in range(12)}

    print("\n" + "=" * 100)
    print("§5b  CRITERION SENSITIVITY: native-area 'small' vs model-space (@1280) 'small'")
    print("=" * 100)
    n_small_img_1280 = int(sum(1 for t, a in zip(train, areas)
                               if len(a) and bool((a * scale2(t) < SMALL_MAX_AREA).any())))
    print(f"  images containing >=1 small instance:")
    print(f"    by NATIVE area  : {n_small_img}/{n} = {100*n_small_img/n:.2f}%   (the canonical SIZE_BINS rule)")
    print(f"    by @1280 area   : {n_small_img_1280}/{n} = {100*n_small_img_1280/n:.2f}%")
    print(f"  instances: native {tot_small}  vs  @1280 {n_small_at1280}  (Δ {n_small_at1280-tot_small:+d})")
    R["criterion_sensitivity"] = dict(images_native=n_small_img, images_1280=n_small_img_1280,
                                      inst_native=tot_small, inst_1280=n_small_at1280)

    R["argv"] = dict(SMALL_MAX_AREA=SMALL_MAX_AREA, BATCH=BATCH, EPOCHS=EPOCHS,
                     BUFFER_MAX=BUFFER_MAX, N_EP_SIM=N_EP, SEED=SEED)
    (out_dir / "p0_results.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=str),
                                            encoding="utf-8")
    print(f"\n[json] {out_dir / 'p0_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
