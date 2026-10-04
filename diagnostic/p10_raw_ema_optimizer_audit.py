"""p10_audit.py — P10 zero-GPU forensic audit: optimizer / EMA / initialization dynamics.

只读。不做 forward / backward / 训练 / 推理 / 验证。
只 torch.load(weights_only=False) 读取已存在的 epoch*.pt 快照并做 CPU 张量代数。

用法:
  python -X utf8 diagnostic/p10_raw_ema_optimizer_audit.py --run \
      --weights-dir runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights \
      --out diagnostic/p10_out
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLASSIFIER_KEY_HINT = "cv3"  # Detect head: model.<i>.cv3.<j>.<k>


def _ln(x: float) -> float:
    return float(np.log(x))


def theoretical_bias_init(nc: int, strides=(8, 16, 32)):
    """head.py:144  b = log(5 / nc / (640 / s) ** 2)"""
    return {int(s): _ln(5 / nc / (640 / s) ** 2) for s in strides}


def cos_schedule(epoch: int, epochs: int, lrf: float) -> float:
    """ultralytics trainer.lf: ((1 - cos(x*pi/epochs)) / 2) * (lrf - 1) + 1"""
    return ((1 - math.cos(epoch * math.pi / epochs)) / 2) * (lrf - 1) + 1


def ema_decay(updates: int, decay=0.9999, tau=2000) -> float:
    return decay * (1 - math.exp(-updates / tau))


def _group_names(model, bn_types):
    """Reproduce build_optimizer's grouping to map global optimizer index -> param name."""
    g = [[], [], []]  # g[0]=weights(with decay), g[1]=bn, g[2]=bias
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:
                g[2].append(fullname)
            elif isinstance(module, bn_types):
                g[1].append(fullname)
            else:
                g[0].append(fullname)
    # optimizer param order = group0(g[2]) + group1(g[0]) + group2(g[1])
    ordered = g[2] + g[0] + g[1]
    return {i: n for i, n in enumerate(ordered)}, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--out", default="diagnostic/p10_out")
    ap.add_argument("--pattern", default="epoch*.pt")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--pretrained", default="yolo11m.pt")
    ap.add_argument("--nc", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr0", type=float, default=0.005)
    ap.add_argument("--lrf", type=float, default=0.01)
    A = ap.parse_args()

    wdir = (ROOT / A.weights_dir).resolve() if not Path(A.weights_dir).is_absolute() else Path(A.weights_dir)
    out = (ROOT / A.out).resolve() if not Path(A.out).is_absolute() else Path(A.out)

    snaps = []
    for p in sorted(wdir.glob(A.pattern)):
        m = re.search(r"epoch(\d+)", p.stem)
        if m:
            snaps.append((int(m.group(1)), p))
    snaps.sort()
    print(f"[snap] {len(snaps)} snapshots in {wdir}")

    if A.check_only:
        print("[check] snapshots:", [e for e, _ in snaps])
        print("[check] theoretical bias init (nc=%d):" % A.nc, theoretical_bias_init(A.nc))
        return

    import torch
    import torch.nn as nn

    bn_types = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    opt_rows = []
    dir_rows = []
    keys_logged = False
    reg_pretrained = None

    # --- reference: pretrained yolo11m.pt head (nc=80) -------------------------
    pre_path = ROOT / A.pretrained
    if pre_path.exists():
        pc = torch.load(pre_path, map_location="cpu", weights_only=False)
        pm = pc.get("model")
        sd = pm.state_dict() if hasattr(pm, "state_dict") else {}
        cls_keys = sorted([k for k in sd if ".cv3." in k and k.endswith((".weight", ".bias"))])
        reg_pretrained = {"path": str(pre_path), "n_cls_keys": len(cls_keys),
                          "sample": cls_keys[:6],
                          "shapes": {k: list(sd[k].shape) for k in cls_keys[:6]}}
        del pc, pm, sd

    for ep, path in snaps:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if not keys_logged:
            print("[ckpt] top-level keys:", sorted(ck.keys()))
            print("[ckpt] version:", ck.get("version"), "| epoch:", ck.get("epoch"),
                  "| updates:", ck.get("updates"))
            keys_logged = True

        ema_mod = ck["ema"]
        sd = ema_mod.state_dict()

        if "model" not in ck or ck.get("model") is None:
            raw_present = False
        else:
            raw_present = True

        # classifier convs: keys of the form model.<L>.cv3.<j>.<k>.{weight,bias}
        cls_w_keys = sorted([k for k in sd if re.search(r"\.cv3\.\d+\.\d+\.weight$", k)])
        cls_b_keys = sorted([k for k in sd if re.search(r"\.cv3\.\d+\.\d+\.bias$", k)])

        # -------- W / b stats per stride branch --------------------------------
        per_branch = {}
        for wk in cls_w_keys:
            bk = wk[:-len("weight")] + "bias"
            W = sd[wk].float().numpy().astype(np.float64).reshape(sd[wk].shape[0], -1)  # (nc, C)
            b = sd[bk].float().numpy().astype(np.float64).reshape(-1)  # (nc,)
            Wm = W.mean(axis=0)
            Wc = W - Wm
            per_branch[wk] = dict(
                W_rows=W.copy(), b=b.copy(),
                W_mean_norm=float(np.linalg.norm(Wm)),
                W_centered_norm=float(np.linalg.norm(Wc)),
                W_fro=float(np.linalg.norm(W)),
                b_mean=float(b.mean()),
                b_std=float(b.std()),
            )

        rows.append(dict(epoch=ep, updates=int(ck.get("updates", -1)),
                         raw_model_present=raw_present,
                         **{f"{k}|{m}": v[m] for k, v in per_branch.items()
                            for m in ("W_mean_norm", "W_centered_norm", "W_fro", "b_mean", "b_std")}))

        dir_rows.append(dict(epoch=ep, **{k: v["W_rows"] for k, v in per_branch.items()}))

        # -------- optimizer state -------------------------------------------------
        opt = ck.get("optimizer")
        if opt is not None:
            idx2name, gs = _group_names(ema_mod, bn_types)
            st = opt["state"]
            entry = dict(epoch=ep,
                         n_groups=len(opt["param_groups"]),
                         group_lrs=[float(pg.get("lr", float("nan"))) for pg in opt["param_groups"]],
                         group_moms=[float(pg.get("momentum", float("nan"))) for pg in opt["param_groups"]],
                         group_wd=[float(pg.get("weight_decay", float("nan"))) for pg in opt["param_groups"]],
                         group_sizes=[len(pg["params"]) for pg in opt["param_groups"]],
                         n_state=int(len(st)))
            # find classifier weight/bias global indices
            for wk in cls_w_keys:
                bk = wk[:-len("weight")] + "bias"
                for key, nm in ((wk, "W"), (bk, "b")):
                    gi = [i for i, n in idx2name.items() if n == key]
                    if not gi:
                        entry[f"{key}|{nm}|status"] = "name_not_found"
                        continue
                    gi = gi[0]
                    s = st.get(gi)
                    if s is None:
                        entry[f"{key}|{nm}|status"] = "no_state"
                        continue
                    mb = s.get("momentum_buffer")
                    if mb is None:
                        entry[f"{key}|{nm}|status"] = "no_momentum_buffer"
                        continue
                    mb = mb.float().numpy().astype(np.float64)
                    if nm == "W":
                        mb = mb.reshape(mb.shape[0], -1)
                        m_mean = mb.mean(axis=0)
                        entry[f"{key}|M_mean_norm"] = float(np.linalg.norm(m_mean))
                        entry[f"{key}|M_centered_norm"] = float(np.linalg.norm(mb - m_mean))
                        entry[f"{key}|M_fro"] = float(np.linalg.norm(mb))
                    else:
                        entry[f"{key}|M_b_mean"] = float(mb.mean())
                        entry[f"{key}|M_b_fro"] = float(np.linalg.norm(mb))
            opt_rows.append(entry)

        del ck, ema_mod, sd
        print(f"  ep{ep:3d}  updates={rows[-1]['updates']:5d}  raw_model={raw_present}")

    # ------------------------------------------------------------------ summary
    epochs = [r["epoch"] for r in rows]
    ups = [r["updates"] for r in rows]

    # pick the s=8 branch (model.<L>.cv3.0.*) as primary
    key0 = sorted(rows[0].keys())
    br_w = [k for k in key0 if k.endswith("|W_mean_norm")]
    # primary = first (cv3.0 = stride 8)
    prim = br_w[0].split("|")[0]
    print(f"\n[primary branch] {prim}")

    Wm = np.array([r[f"{prim}|W_mean_norm"] for r in rows])
    Wc = np.array([r[f"{prim}|W_centered_norm"] for r in rows])
    Wf = np.array([r[f"{prim}|W_fro"] for r in rows])
    bm = np.array([r[f"{prim}|b_mean"] for r in rows])
    bs = np.array([r[f"{prim}|b_std"] for r in rows])

    dWm = np.abs(np.diff(Wm))
    total_movement = float(np.sum(dWm))
    cum = np.concatenate([[0.0], np.cumsum(dWm)])
    frac = cum / total_movement if total_movement > 0 else np.zeros_like(cum)

    dec = np.array([ema_decay(u) for u in ups])
    lr_sched = np.array([cos_schedule(e, A.epochs, A.lrf) for e in epochs])

    tbl = []
    for i, e in enumerate(epochs):
        tbl.append(dict(epoch=e, updates=ups[i], ema_decay=dec[i],
                        lr_factor=lr_sched[i], lr_eff=A.lr0 * lr_sched[i],
                        W_mean_norm=Wm[i], W_centered_norm=Wc[i], W_fro=Wf[i],
                        b_mean=bm[i], b_std=bs[i],
                        dWmean_from_prev=float(dWm[i - 1]) if i else 0.0,
                        cum_frac_Wmean=float(frac[i])))

    # -------- direction audit -------------------------------------------------
    ref = dir_rows[0]
    def _wm(mat):  # (nc, C) -> mean row vector
        return mat.mean(axis=0)

    dird = []
    for r in dir_rows:
        e = r["epoch"]
        row = {"epoch": e}
        for k in [k for k in r if k != "epoch"]:
            A0 = _wm(ref[k]); A1 = _wm(r[k])
            n0 = np.linalg.norm(A0); n1 = np.linalg.norm(A1)
            row[f"{k}|cos_Wmean_vs_ep0"] = float(A0 @ A1 / (n0 * n1 + 1e-30))
            # per-row direction cos to ep0 W_mean
            cs = (r[k] @ A0) / (np.linalg.norm(r[k], axis=1) * n0 + 1e-30)
            row[f"{k}|rowcos_to_ep0_Wmean_mean"] = float(cs.mean())
        dird.append(row)

    # consecutive rotation
    rot = []
    for i in range(1, len(dir_rows)):
        e0, e1 = dir_rows[i - 1], dir_rows[i]
        row = {"epoch": e1["epoch"], "prev": e0["epoch"]}
        for k in [k for k in e1 if k != "epoch"]:
            a = _wm(e0[k]); b = _wm(e1[k])
            row[f"{k}|cos_Wmean_consecutive"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        rot.append(row)

    # -------- optimizer summary ----------------------------------------------
    osum = []
    for o in opt_rows:
        d = {"epoch": o["epoch"], "group_lrs": o["group_lrs"], "group_moms": o["group_moms"],
             "group_wd": o["group_wd"], "group_sizes": o["group_sizes"], "n_state": o["n_state"]}
        for k, v in o.items():
            if "|M_" in k or k.endswith("|status"):
                d[k] = v
        osum.append(d)

    meta = dict(
        weights_dir=str(wdir), n_snapshots=len(snaps),
        epochs=epochs, updates=ups,
        theoretical_bias_init=theoretical_bias_init(A.nc),
        pretrained_head=reg_pretrained,
        ema_formula="decay(t)=0.9999*(1-exp(-t/2000))  [torch_utils.py:546]",
        ema_update_site="Trainer.optimizer_step  trainer.py:595  -> 1 update per optimizer step",
        ckpt_semantics="save_model trainer.py:520-537 : {'model': None, 'ema': fp16 EMA, 'optimizer': state_dict}",
        lr_sched="lf(epoch)=((1-cos(epoch*pi/300))/2)*(lrf-1)+1 ; lr=lr0*lf  (cos_lr=True)",
        primary_branch=prim, total_Wmean_movement=total_movement,
    )

    (out / "p10_summary.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "p10_trajectory.json").write_text(json.dumps(tbl, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "p10_direction.json").write_text(json.dumps(dird, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "p10_rotation.json").write_text(json.dumps(rot, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "p10_optimizer.json").write_text(json.dumps(osum, indent=2, ensure_ascii=False), encoding="utf-8")

    # -------- console table ---------------------------------------------------
    print("\n ep  updates  ema_decay   lr_eff    |Wmean|  |Wcent|   dWmean  cum%   b_mean  b_std")
    for r in tbl:
        print(f"{r['epoch']:3d} {r['updates']:7d}  {r['ema_decay']:.4f}  {r['lr_eff']:.6f}  "
              f"{r['W_mean_norm']:7.4f} {r['W_centered_norm']:7.4f} {r['dWmean_from_prev']:8.5f} "
              f"{100*r['cum_frac_Wmean']:5.1f}%  {r['b_mean']:+7.4f} {r['b_std']:.4f}")

    print("\n[dir] cos(W_mean_ep, W_mean_ep0):")
    for r in dird:
        print(f"  ep{r['epoch']:3d}  " + "  ".join(f"{k.split('|')[0][-8:]}:{v:+.5f}"
              for k, v in r.items() if k.endswith("cos_Wmean_vs_ep0")))

    print("\n[summary]")
    print(f"  total |dWmean| over {len(epochs)} snapshots = {total_movement:.5f}")
    for r in tbl:
        if r["epoch"] in (0, 10, 20, 30, 50, 100, 150, 200, 290):
            print(f"  ep{r['epoch']:3d}: cum_frac={r['cum_frac_Wmean']:.4f}")


if __name__ == "__main__":
    main()
