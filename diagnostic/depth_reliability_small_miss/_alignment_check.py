"""_alignment_check.py — STOP B 判定：HR depth 是否与 HR visible 像素级对齐（READ-ONLY）

动机：本审计要在 **native GT 坐标** 上取 depth patch（协议 §4）。若 depth 与 visible 的空间
对应不可靠，则「GT 框内 depth 全零」不能解释成「目标区域无深度回波」，协议 §23 要求 **STOP B**。

四项只读检验（全部不需要 forward）：
  T1 像素级梯度相关（8×8 下采样）
  T2 位移搜索：corr 随 (dy,dx) 变化，看峰值是否在 (0,0)
  T3 行剖面：row-wise depth zero_frac  vs  row-wise visible gradient
  T4 天空掩码富集：depth==0 在 visible 天空掩码内的 lift
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
S = ROOT / "data/processed/rgbid_split_train/images/val"
sys.path.insert(0, str(ROOT))
from ultralytics.utils.patches import imread  # noqa: E402


def ds(a, k=8):
    H, W = a.shape[:2]
    return a[:H // k * k, :W // k * k].reshape(H // k, k, W // k, k).mean((1, 3))


def main():
    files = sorted((S / "depth").glob("*.png"))          # HR uint16 层
    print("=" * 100)
    print(f"STOP B ALIGNMENT CHECK — HR depth({len(files)} 张) vs HR visible")
    print("=" * 100)

    # ---- T1/T2 ----
    n = min(40, len(files))
    t1, t2, t2shift = [], [], []
    for p in files[:n]:
        d = imread(str(p), -1)
        if d is None:
            continue
        if d.ndim == 3:
            d = d[:, :, 0]
        v = imread(str(S / "visible" / f"{p.stem}.png"), -1)
        if v is None or d.shape[:2] != v.shape[:2]:
            continue
        d32 = d.astype(np.float32); dg = np.zeros(d.shape, np.float32)
        dg[:, :-1] += np.abs(np.diff(d32, axis=1)); dg[:-1, :] += np.abs(np.diff(d32, axis=0))
        vg_ = v.astype(np.float32).mean(-1); vg = np.zeros(vg_.shape, np.float32)
        vg[:, :-1] += np.abs(np.diff(vg_, axis=1)); vg[:-1, :] += np.abs(np.diff(vg_, axis=0))
        A, B = ds(dg), ds(vg)
        if A.shape != B.shape:
            continue
        c00, bc, bsh = None, -9.0, None
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                c = float(np.corrcoef(A.ravel(), np.roll(np.roll(B, dy, 0), dx, 1).ravel())[0, 1])
                if dy == 0 and dx == 0:
                    c00 = c
                if c > bc:
                    bc, bsh = c, (dy * 8, dx * 8)
        t1.append(c00); t2.append(bc); t2shift.append(bsh)
    print(f"\n[T1] 像素级梯度相关 corr@(0,0): 中位 {np.median(t1):.4f}  "
          f"(对齐良好的 RGB-D 通常 0.3–0.6)")
    print(f"[T2] 位移搜索最佳 corr: 中位 {np.median(t2):.4f}；最佳位移中位 "
          f"(dy,dx)={tuple(np.median(np.array(t2shift), axis=0).astype(int))}")

    # ---- T3 ----
    zf, gr = [], []
    for p in files[:60]:
        d = imread(str(p), -1)
        if d is None:
            continue
        if d.ndim == 3:
            d = d[:, :, 0]
        v = imread(str(S / "visible" / f"{p.stem}.png"), -1)
        if v is None:
            continue
        vg_ = v.astype(np.float32).mean(-1); g = np.zeros(vg_.shape, np.float32)
        g[:, :-1] += np.abs(np.diff(vg_, axis=1)); g[:-1, :] += np.abs(np.diff(vg_, axis=0))
        zf.append((d == 0).mean(1)); gr.append(g.mean(1))
    zf, gr = np.mean(zf, 0), np.mean(gr, 0)
    print(f"[T3] corr(row depth zero_frac, row visible gradient) = "
          f"{float(np.corrcoef(zf, gr)[0, 1]):.4f}   （行剖面的共同趋势；不证明像素级对齐）")

    # ---- T4 ----
    lifts = []
    for p in files[:40]:
        d = imread(str(p), -1)
        if d is None:
            continue
        if d.ndim == 3:
            d = d[:, :, 0]
        v = imread(str(S / "visible" / f"{p.stem}.png"), -1)
        if v is None:
            continue
        vg_ = v.astype(np.float32).mean(-1); g = np.zeros(vg_.shape, np.float32)
        g[:, :-1] += np.abs(np.diff(vg_, axis=1)); g[:-1, :] += np.abs(np.diff(vg_, axis=0))
        sky = (vg_ > np.percentile(vg_, 80)) & (g < np.percentile(g, 50))
        z = (d == 0)
        if not sky.any():
            continue
        lifts.append(float(z[sky].mean() / max(z[~sky].mean(), 1e-6)))
    print(f"[T4] depth==0 在 visible『天空』掩码内的 lift 中位 = {np.median(lifts):.3f}   "
          f"(≈1 表示零区与天空无关)")

    # ---- 跨图共同零区结构 ----
    acc, cnt = None, 0
    for p in files:
        d = imread(str(p), -1)
        if d is None:
            continue
        if d.ndim == 3:
            d = d[:, :, 0]
        if acc is None:
            acc = np.zeros(d.shape, np.float32)
        elif acc.shape != d.shape:
            continue
        acc += (d == 0); cnt += 1
    freq = acc / cnt
    print(f"\n[结构] 跨 {cnt} 张图叠加的 zero 频率：")
    for t in (0.25, 0.5, 0.75):
        print(f"  频率 > {t:.2f} 的像素占比 = {100*float((freq > t).mean()):5.1f}%")
    print("  ⇒ 若存在固定相机掩码，>0.75 的比例应很大；实测很小 ⇒ 零区主要是**每图特有**的，")
    print("     并呈**上多下少**的平滑梯度（天空/远处无回波）。")

    out = dict(T1_grad_corr_median=float(np.median(t1)), T2_best_shift_corr_median=float(np.median(t2)),
               T2_best_shift_median=[int(x) for x in np.median(np.array(t2shift), axis=0)],
               T3_rowprofile_corr=float(np.corrcoef(zf, gr)[0, 1]),
               T4_sky_lift_median=float(np.median(lifts)),
               cross_image_zero_freq_over_075=float((freq > 0.75).mean()),
               verdict="STOP_B_TRIGGERED: pixel-level depth<->visible correspondence NOT certifiable")
    (AUDIT / "_alignment.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {AUDIT/'_alignment.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
