"""可见光集的近重复检查（test / test_pre / train / val 之间）。

严格只读。先用 16x16 aHash 粗筛（会大量碰撞，不可作为判据），
再用 64x64 零均值归一化余弦相似度做**强判据**，只把 cosine > 0.95 记为近重复。
目的：确认测试集与 train/val 之间是否存在内容级重复（泄漏 / 复用）。
"""
from __future__ import annotations

import collections
import glob
import json
import os
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SETS = {
    "test": ROOT / "data/raw/test/visible",
    "test_pre": ROOT / "data/raw/test初赛/visible",
    "train": ROOT / "data/raw/train/visible",
    "split_val": ROOT / "data/processed/rgbid_split/images/val/visible",
}
N = 64


def job(t):
    p, s = t
    r = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if r is None:
        return (s, p.stem, None)
    g = cv2.resize(r, (N, N), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    g -= g.mean()
    n = np.linalg.norm(g)
    if n > 0:
        g /= n
    return (s, p.stem, g)


def main():
    jobs = []
    for s, d in SETS.items():
        jobs += [(Path(p), s) for p in sorted(glob.glob(str(d / "*")))]
    with Pool(16) as pool:
        res = pool.map(job, jobs, chunksize=16)
    ok = [r for r in res if r[2] is not None]
    print("files:", len(res), "decoded:", len(ok))

    keys = [(s, stem) for s, stem, _ in ok]
    X = np.stack([g for _, _, g in ok])                       # [M, N*N]
    setidx = np.array([k[0] for k in keys])
    test_m = np.isin(setidx, ["test", "test_pre"])
    train_m = np.isin(setidx, ["train", "split_val"])
    T, R = X[test_m], X[train_m]
    tk, rk = [k for k, m in zip(keys, test_m) if m], [k for k, m in zip(keys, train_m) if m]

    S = T @ R.T                                                # 余弦相似度
    print("cosine matrix:", S.shape)
    for thr in [0.99, 0.95, 0.90, 0.80]:
        print(f"  pairs with cosine > {thr}: {(S > thr).sum()}")
    for thr in [0.99, 0.95]:
        pairs = np.argwhere(S > thr)
        if len(pairs):
            print(f"  --- top pairs at >{thr} ---")
            order = np.argsort(-S[pairs[:, 0], pairs[:, 1]])[:10]
            for i in order:
                a, b = pairs[i]
                print(f"    cos={S[a,b]:.4f} {tk[a][0]}/{tk[a][1]}  <->  {rk[b][0]}/{rk[b][1]}")

    # 最相似对手的相似度分布（每张 test 图取与 train 的最大余弦）
    mx = S.max(axis=1)
    print("max-cosine-to-train distribution over test images:",
          {f"p{q}": round(float(np.percentile(mx, q)), 4) for q in [50, 90, 95, 99, 100]})
    # 对照：train 内部自相似的分布（去掉自身）
    Srr = R @ R.T
    np.fill_diagonal(Srr, -1)
    mxr = Srr.max(axis=1)
    print("max-cosine-within-train distribution (excl. self):",
          {f"p{q}": round(float(np.percentile(mxr, q)), 4) for q in [50, 90, 95, 99, 100]})

    (Path(__file__).resolve().parent / "near_dup.json").write_text(json.dumps({
        "n": len(res), "decoded": len(ok),
        "counts": {str(t): int((S > t).sum()) for t in [0.99, 0.95, 0.90, 0.80]},
        "test_max_cos_to_train": {f"p{q}": float(np.percentile(mx, q)) for q in [50, 90, 95, 99, 100]},
        "train_max_cos_within": {f"p{q}": float(np.percentile(mxr, q)) for q in [50, 90, 95, 99, 100]},
    }, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
