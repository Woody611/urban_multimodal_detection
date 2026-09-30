"""_v10_stages.py — cv3 逐层 trajectory：P3 → S1 → S2(=H)。只读。

只读。只碰 train/val。0 训练 / 0 改模型-head-loss-assigner-增强 / 0 核心文件改动 / 0 碰 test-submission-online。

阶段定义（取自代码，非假设）：
  cv3[0][0] = Sequential(DWConv s1, Conv s1)   → S1
  cv3[0][1] = Sequential(DWConv s1, Conv s1)   → S2  == H（cls Conv 的输入）
  cv3[0][2] = Conv2d(256, nc, 1)               → 分类头，**不计入 representation stage**
⇒ 只有 3 个可观测 stage。

同一次 forward 同时取三层；GT-center 采样与 V7 完全一致（单 cell、输入空间像素 /8）。
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "diagnostic/small_object_cause_v2"))

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, yaml_model_load  # noqa: E402
from ultralytics.utils import yaml_load  # noqa: E402

OUT = Path(__file__).resolve().parent
CKPT = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
V2 = ROOT / "diagnostic/small_object_cause_v2"
SEED = 20260929
N_TRAIN_IMG = 300
SMALL = 32.0


def fit_lr(X, y, lam=1.0, iters=12):
    n, d = X.shape
    Xa = np.hstack([X, np.ones((n, 1))]); th = np.zeros(d + 1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xa @ th, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-6)
        H = Xa.T @ (Xa * W[:, None]) + lam * np.eye(d + 1)
        try:
            step = np.linalg.solve(H, Xa.T @ (p - y) + lam * th)
        except np.linalg.LinAlgError:
            break
        th -= step
        if np.abs(step).max() < 1e-8:
            break
    return th[:d], float(th[d])


def auc(y, s):
    o = np.argsort(s); y = y[o]; n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.arange(1, len(y) + 1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cv_eval(X, y, folds=5, seed=0):
    rng = np.random.default_rng(seed); parts = []
    for c in (0, 1):
        ci = np.where(y == c)[0].copy(); rng.shuffle(ci)
        parts.append(np.array_split(ci, folds))
    A, U = [], []
    for f in range(folds):
        te = np.r_[parts[0][f], parts[1][f]]; tr = np.setdiff1d(np.arange(len(y)), te)
        w, b = fit_lr(X[tr], y[tr]); s = X[te] @ w + b
        pr = (s > 0).astype(int)
        A.append(np.mean([np.mean(pr[y[te] == c] == c) for c in (0, 1)]))
        U.append(auc(y[te], s))
    return float(np.mean(A)), float(np.std(A)), float(np.mean(U)), float(np.std(U))


def cliffs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (sum((x > b).sum() for x in a) - sum((x < b).sum() for x in a)) / (len(a) * len(b))


def boot(a, b, n=2000):
    rng = np.random.default_rng(0); a, b = np.asarray(a, float), np.asarray(b, float)
    d = [np.median(rng.choice(a, len(a), True)) - np.median(rng.choice(b, len(b), True))
         for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    cfg = yaml_model_load(str(ROOT / "configs/yolo11m_sepstem.yaml"))
    model = DetectionModel(cfg, nc=12, verbose=False)
    w, _ = attempt_load_one_weight(str(CKPT)); model.load(w)
    model.model[-1].stride = model.stride
    model.eval(); model.model[-1].train()
    d = model.model[-1]
    cur = {}
    for tag, mod in (("P3", model.model[23]), ("S1", d.cv3[0][0]), ("S2", d.cv3[0][1])):
        def mk(t):
            def h(m, i, o):
                if torch.is_tensor(o):
                    cur[t] = o.detach()
            return h
        mod.register_forward_hook(mk(tag))
    print("hooks: P3=layer23 out, S1=cv3[0][0] out, S2=cv3[0][1] out (=H)")

    data = yaml_load(str(ROOT / "data/processed/rgbid_split_train/dataset.yaml"))
    rows = []
    for dom, tr in (("T1", True), ("V1", False)):
        ov = dict(imgsz=1280, task="detect", rect=False, cache=False, single_cls=False,
                  classes=None, fraction=1.0, channels=5, use_simotm="RGBID",
                  object_scale_aug=False, mosaic=0.0, mixup=0.0, copy_paste=0.0,
                  degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
                  flipud=0.0, fliplr=0.5, augment=False)
        ds = build_yolo_dataset(get_cfg(overrides=ov),
                                str((Path(data["path"]) / (data["train"] if tr else data["val"])).resolve()),
                                8, data, mode="val", use_simotm="RGBID",
                                pairs_rgb_ir=["visible", "infrared"],
                                pairs_rgb_depth=["visible", "depth"])
        idxs = (list(range(0, len(ds), max(1, len(ds) // N_TRAIN_IMG)))[:N_TRAIN_IMG]
                if dom == "T1" else range(len(ds)))
        n0 = len(rows)
        for i in idxs:
            random.seed(SEED + i); np.random.seed(SEED + i)
            lab = ds[i]; bb = lab.get("bboxes"); cl = lab.get("cls")
            if bb is None or len(bb) == 0:
                continue
            img = lab["img"].float().div(255.0)
            img = img.unsqueeze(0) if img.ndim == 3 else img
            cur.clear()
            with torch.no_grad():
                model(img)
            if len(cur) < 3:
                continue
            stem = Path(ds.im_files[i]).stem
            for j in range(len(bb)):
                cx, cy = float(bb[j][0]) * 1280, float(bb[j][1]) * 1280
                wpx, hpx = float(bb[j][2]) * 1280, float(bb[j][3]) * 1280
                r = dict(domain=dom, stem=stem, gt=j, key=f"{stem}#{j}",
                         cls=int(cl[j]), sq=float(np.sqrt(max(wpx * hpx, 0))))
                for tag in ("P3", "S1", "S2"):
                    t = cur[tag]
                    hh, ww = t.shape[2:]
                    sc = 1280.0 / ww
                    x = int(min(max(cx / sc, 0), ww - 1)); y = int(min(max(cy / sc, 0), hh - 1))
                    r[tag] = t[0, :, y, x].numpy().astype(np.float32).tolist()
                rows.append(r)
        print(f"  [{dom}] {len(rows)-n0} GT")
    (OUT / "_v10_stages.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {OUT/'_v10_stages.json'} ({len(rows)} GT)")

    # ---------------- 分析 ----------------
    v3 = json.loads((V2 / "_v3_analysis.json").read_text(encoding="utf-8"))
    E1 = set(v3["E1"]); CTRL = set(v[0] for v in v3["controls"].values())
    v2rec = {f"{r['image_id']}#{r['gt_id']}": r for r in
             json.loads((V2 / "_v2_records.json").read_text(encoding="utf-8"))["records"]}
    from official_eval import (load_split, read_pred_txt, apply_max_boxes,
                               norm_xywh_to_xyxy, box_iou_np, MAX_BOXES_PER_IMAGE)
    succ = {}
    for _i, s_, w_, h_, gb, gc in load_split(
            ROOT / "data/processed/rgbid_split_train/images/val/visible",
            ROOT / "data/processed/rgbid_split_train/labels/val/visible")[0]:
        if gb is None or len(gb) == 0:
            continue
        _, pr, _ = read_pred_txt(ROOT / f"diagnostic/sepstem_clahe/best_full/results/{s_}.txt")
        pr = apply_max_boxes(pr, MAX_BOXES_PER_IMAGE)
        pb, pc = (norm_xywh_to_xyxy(pr, w_, h_) if len(pr) else
                  (np.zeros((0, 4), np.float32), np.zeros(0, int)))
        for j in range(len(gb)):
            sm = np.where(pc == gc[j])[0] if len(pc) else np.zeros(0, int)
            succ[f"{s_}#{j}"] = (float(box_iou_np(gb[j:j+1], pb[sm])[0].max()) if len(sm) else 0.0) >= 0.5
    for r in rows:
        if r["domain"] == "T1":
            r["grp"] = "T_success" if r["sq"] < SMALL else None
        else:
            k = r["key"]
            r["grp"] = ("G1" if k in E1 else "G2" if k in CTRL else
                        "G4" if (k in v2rec and k not in E1 and k not in CTRL
                                  and v2rec[k]["native_area"] < 1024 and succ.get(k)) else None)

    print("\n" + "=" * 108)
    print("§10 核心表：逐 stage 的 distribution shift（G1 vs train-success small）")
    print("=" * 108)
    ST = ("P3", "S1", "S2")
    G1 = [r for r in rows if r["grp"] == "G1"]
    TS = [r for r in rows if r["grp"] == "T_success"]
    G4 = [r for r in rows if r["grp"] == "G4"]
    print(f"  n: G1={len(G1)}  T_success={len(TS)}  G4={len(G4)}")
    print(f"\n  {'Stage':<6}{'|G1| med':>10}{'|TS| med':>10}{'cos→μ_TS':>11}{'kNN5→TS':>10}"
          f"{'AUC(G1/TS)':>12}{'balAcc':>10}{'CliffΔ':>9}{'ΔCI95':>20}")
    traj_ts, traj_g4 = {}, {}
    for st in ST:
        XG = np.stack([np.array(r[st], np.float64) for r in G1])
        XT = np.stack([np.array(r[st], np.float64) for r in TS])
        X4 = np.stack([np.array(r[st], np.float64) for r in G4])
        # 全局标准化（同一 stage 内，用 TS+G1+G4 合并）
        A = np.vstack([XG, XT, X4]); mu, sd = A.mean(0), A.std(0) + 1e-8
        ZG, ZT, Z4 = ((X - mu) / sd for X in (XG, XT, X4))
        muT = ZT.mean(0); muT /= np.linalg.norm(muT)
        cosG = float(np.median(ZG @ muT / (np.linalg.norm(ZG, axis=1) + 1e-12)))
        cosT = float(np.median(ZT @ muT / (np.linalg.norm(ZT, axis=1) + 1e-12)))
        nT = ZT / (np.linalg.norm(ZT, axis=1, keepdims=True) + 1e-12)
        nG = ZG / (np.linalg.norm(ZG, axis=1, keepdims=True) + 1e-12)
        D = np.linalg.norm(nG[:, None, :] - nT[None, :, :], axis=2); D.sort(1)
        knn = float(np.median(D[:, :5].mean(1)))
        acc, accs, aud, auds = cv_eval(np.vstack([ZG, ZT]), np.r_[np.ones(len(ZG)), np.zeros(len(ZT))])
        w_, b_ = fit_lr(np.vstack([ZG, ZT]), np.r_[np.ones(len(ZG)), np.zeros(len(ZT))])
        scG, scT = ZG @ w_, ZT @ w_
        lo, hi = boot(scG, scT)
        traj_ts[st] = aud
        acc4, _, aud4, _ = cv_eval(np.vstack([ZG, Z4]), np.r_[np.ones(len(ZG)), np.zeros(len(Z4))])
        traj_g4[st] = aud4
        print(f"  {st:<6}{np.median(np.linalg.norm(XG)):>10.2f}{np.median(np.linalg.norm(XT)):>10.2f}"
              f"{cosG:>11.4f}{knn:>10.4f}{aud:>12.4f}{acc:>10.4f}{cliffs(scG,scT):>9.3f}"
              f"{f'[{lo:+.2f},{hi:+.2f}]':>20}")

    print("\n  §7 AUC trajectory")
    print(f"  {'对比':<20}{'P3':>9}{'S1':>9}{'S2':>9}")
    print(f"  {'G1 vs train-succ':<20}{traj_ts['P3']:>9.4f}{traj_ts['S1']:>9.4f}{traj_ts['S2']:>9.4f}")
    print(f"  {'G1 vs G4':<20}{traj_g4['P3']:>9.4f}{traj_g4['S1']:>9.4f}{traj_g4['S2']:>9.4f}")

    print("\n  §9 三组控制：cos(v, μ_Tsuccess)（同 stage 同一 μ）")
    for st in ST:
        A = np.vstack([np.stack([np.array(r[st], np.float64) for r in G1]),
                       np.stack([np.array(r[st], np.float64) for r in TS]),
                       np.stack([np.array(r[st], np.float64) for r in G4])])
        mu, sd = A.mean(0), A.std(0) + 1e-8
        ZT = (np.stack([np.array(r[st], np.float64) for r in TS]) - mu) / sd
        muT = ZT.mean(0); muT /= np.linalg.norm(muT)
        out = []
        for g in (G1, G4):
            Z = (np.stack([np.array(r[st], np.float64) for r in g]) - mu) / sd
            out.append(float(np.median(Z @ muT / (np.linalg.norm(Z, axis=1) + 1e-12))))
        print(f"    {st}: G1 rec={out[0]:+.4f}   G4 rec={out[1]:+.4f}   (train-succ 自身 = 0.8007 参照 V7)")

    print("\n  §11 分离度增量（AUC 差，G1 vs train-succ）")
    print(f"    Δ(P3→S1) = {traj_ts['S1']-traj_ts['P3']:+.4f}   Δ(S1→S2) = {traj_ts['S2']-traj_ts['S1']:+.4f}")


if __name__ == "__main__":
    main()
