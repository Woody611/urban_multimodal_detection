"""量化 test 与 train/val 的内容重叠程度（只读）。

问题：near_dup_check 发现 test/00000272 <-> train/00000273 这类**相邻帧**配对，
且 train 内部 p90 的最大余弦 = 1.0。需要判断：
  (a) 这是源数据集本身含重复帧（自重复），还是
  (b) train/test 划分真的在序列内切帧（跨集近重复）。
两者对「测试集有效样本量」与「域偏移幅度」的含义完全不同。
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
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


def clip_of(stem):
    """从文件名解析「序列号」：末段数字。"""
    m = re.findall(r"\d+", stem)
    return int(m[-1]) if m else -1


def main():
    jobs = []
    for s, d in SETS.items():
        jobs += [(Path(p), s) for p in sorted(glob.glob(str(d / "*")))]
    with Pool(16) as pool:
        res = pool.map(job, jobs, chunksize=16)
    ok = [r for r in res if r[2] is not None]
    keys = [(s, stem) for s, stem, _ in ok]
    X = np.stack([g for _, _, g in ok])
    setidx = np.array([k[0] for k in keys])
    print("decoded:", len(ok))

    out = {}
    for grp, members in [("test(+pre)", ["test", "test_pre"]), ("train_raw", ["train"]),
                         ("split_val", ["split_val"])]:
        m = np.isin(setidx, members)
        A = X[m]
        kk = [k for k, mm in zip(keys, m) if mm]
        S = A @ A.T
        np.fill_diagonal(S, -1)
        mx = S.max(axis=1)
        jj = S.argmax(axis=1)
        hi = mx > 0.99
        # 相邻帧判别：末段数字差 <= 3
        adj = 0
        for i in np.where(hi)[0]:
            if abs(clip_of(kk[i][1]) - clip_of(kk[jj[i]][1])) <= 3:
                adj += 1
        out[grp] = {
            "n": len(kk),
            "frac_maxcos_gt0999": float((mx > 0.999).mean()),
            "frac_maxcos_gt099": float(hi.mean()),
            "frac_maxcos_gt095": float((mx > 0.95).mean()),
            "of_which_adjacent_frame_lt3": int(adj),
            "pct_maxcos": {f"p{q}": float(np.percentile(mx, q)) for q in [50, 75, 90, 95, 99]},
        }
        print(f'{grp:12s} n={len(kk):5d} maxcos>0.999 {out[grp]["frac_maxcos_gt0999"]:.3f} '
              f'>0.99 {hi.mean():.3f} >0.95 {(mx>0.95).mean():.3f}  adjacent(<=3) {adj}  '
              f'pct={ {k: round(v,4) for k,v in out[grp]["pct_maxcos"].items()} }')

    # 跨集：test -> train_raw / split_val
    tm = np.isin(setidx, ["test", "test_pre"])
    rm = np.isin(setidx, ["train", "split_val"])
    T, R = X[tm], X[rm]
    tk = [k for k, mm in zip(keys, tm) if mm]
    rk = [k for k, mm in zip(keys, rm) if mm]
    S = T @ R.T
    mx = S.max(axis=1)
    jj = S.argmax(axis=1)
    out["test_to_trainval"] = {
        "n": len(tk),
        "frac_maxcos_gt0999": float((mx > 0.999).mean()),
        "frac_maxcos_gt099": float((mx > 0.99).mean()),
        "frac_maxcos_gt095": float((mx > 0.95).mean()),
        "pct": {f"p{q}": float(np.percentile(mx, q)) for q in [50, 75, 90, 95, 99]},
    }
    print("\ntest -> train/val:",
          {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["test_to_trainval"].items() if k != "pct"},
          {k: round(v, 4) for k, v in out["test_to_trainval"]["pct"].items()})

    # 高相似配对的命名关系
    hi = np.where(mx > 0.99)[0]
    rel = collections.Counter()
    ex = []
    for i in hi:
        a, b = tk[i], rk[jj[i]]
        da, db = clip_of(a[1]), clip_of(b[1])
        rel[f"{a[0]}->{b[0]} |Δframe|={min(abs(da-db),9) if abs(da-db)<10 else '>=10'}"] += 1
        if len(ex) < 12:
            ex.append((round(float(mx[i]), 4), f"{a[0]}/{a[1]}", f"{b[0]}/{b[1]}"))
    print("\n高相似(>0.99)配对的帧号关系分布:", dict(rel.most_common(20)))
    print("\n示例:")
    for e in ex:
        print("   ", e)

    # test 内部分辨率/族 与「是否有 train 近邻」的关系
    fam_of = {}
    for s, stem in tk:
        if stem.startswith("shuming_"):
            fam_of[(s, stem)] = "shuming_*"
        elif stem.startswith("hehe_"):
            fam_of[(s, stem)] = "hehe_*"
        elif "suppl" in stem:
            fam_of[(s, stem)] = "*suppl*"
        elif stem.count("_") == 2:
            fam_of[(s, stem)] = "N_N_N"
        elif stem.count("_") == 0:
            fam_of[(s, stem)] = "N"
        else:
            fam_of[(s, stem)] = "other"
    byfam = collections.defaultdict(lambda: [0, 0])
    for i, k in enumerate(tk):
        f = fam_of[k]
        byfam[f][0] += 1
        if mx[i] > 0.99:
            byfam[f][1] += 1
    print("\ntest 各族「与 train/val 有 >0.99 近邻」的比例:")
    for f, (n, h) in sorted(byfam.items(), key=lambda x: -x[1][0]):
        print(f"   {f:11s} n={n:5d} frac={h/n:.3f}")

    (Path(__file__).resolve().parent / "dup_quantify.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
