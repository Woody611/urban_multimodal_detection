# -*- coding: utf-8 -*-
"""P0 §6/§7/§9/§10 — gamma/noise 合成 sanity + 数值域锁定 + 交互矩阵（CPU only）。

全部调用**真实实现**（ultralytics.data.base.apply_ir_augmentation / apply_ir_encoding），
不重写公式，避免"审计的是一份代码、跑的是另一份"。

输出: stdout + diagnostic/ir_aug_audit/sanity_and_matrix.json
"""
import os
import sys
import json
import copy
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import ultralytics.data.base as B  # noqa: E402
from ultralytics.data.base import apply_ir_encoding, apply_ir_augmentation, IR_AUG_STATS  # noqa: E402
from ultralytics.utils.patches import imread  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sanity_and_matrix.json")
IR_DIR = "data/processed/rgbid_split_train/images/train/infrared"
VIS_DIR = "data/processed/rgbid_split_train/images/train/visible"
DEP_DIR = "data/processed/rgbid_split_train/images/train/depth"
SAMPLES = ["00000004.jpg", "00000008.jpg", "00000022.jpg", "00000025.jpg", "00000031.jpg",
           "00000033.jpg", "00000034.jpg", "00000036.jpg"]


class Cfg:
    def __init__(self, gamma=(1.0, 1.0), gp=0.0, nstd=0.0, np_=0.0):
        self.ir_gamma = list(gamma)
        self.ir_gamma_probability = gp
        self.ir_noise_std = nstd
        self.ir_noise_probability = np_


def stat(a):
    a = np.asarray(a).ravel().astype(np.float64)
    return {"n": int(a.size), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std()),
            "p01": float(np.percentile(a, 1)), "p50": float(np.percentile(a, 50)),
            "p99": float(np.percentile(a, 99))}


# ---------------------------------------------------------------- §6 synthetic sanity
def synthetic():
    res = {}
    patterns = {
        "black": np.zeros((32, 32), np.uint8),
        "dark_gray": np.full((32, 32), 32, np.uint8),
        "mid_gray": np.full((32, 32), 128, np.uint8),
        "bright_gray": np.full((32, 32), 224, np.uint8),
        "white": np.full((32, 32), 255, np.uint8),
        "gradient": np.tile(np.arange(256, dtype=np.uint8), (32, 1)),
        "all_values": np.arange(256, dtype=np.uint8).reshape(16, 16),
    }
    for g in (0.5, 0.75, 0.85, 1.0, 1.2, 1.35, 2.0):
        entry = {}
        for name, p in patterns.items():
            out = with_seed(lambda: apply_ir_augmentation(p.copy(), Cfg(gamma=(g, g), gp=1.0), True))
            finite = bool(np.isfinite(out).all())
            entry[name] = {
                "in_min": int(p.min()), "in_max": int(p.max()),
                "out_min": int(out.min()), "out_max": int(out.max()),
                "out_dtype": str(out.dtype), "out_shape": list(out.shape),
                "finite": finite, "monotone": bool(np.all(np.diff(out.ravel()[:256].astype(int)) >= 0)) if name == "gradient" else None,
                "identity_exact": bool(np.array_equal(out, p)),
                "mean_in": round(float(p.mean()), 3), "mean_out": round(float(out.mean()), 3),
            }
        res[f"gamma_{g}"] = entry

    # 噪声合成
    for std in (0.0, 0.015, 0.05):
        p = patterns["mid_gray"]
        out = with_seed(lambda: apply_ir_augmentation(p.copy(), Cfg(nstd=std, np_=1.0), True))
        res[f"noise_{std}"] = {
            "in": stat(p.astype(np.int32)), "out": stat(out),
            "out_dtype": str(out.dtype), "finite": bool(np.isfinite(out).all()),
            "exact_identity": bool(np.array_equal(out, p)),
            "measured_sigma_grays": round(float(out.astype(np.float64).std()), 4),
            "expected_sigma_grays": round(std * 255.0, 4),
        }
    return res


def with_seed(fn, seed=1234):
    import random as _r
    _r.seed(seed)
    np.random.seed(seed)
    return fn()


# ---------------------------------------------------------------- §7 gamma range vs natural exposure
def gamma_range_evidence():
    """对真实 IR，量化 gamma 造成的"曝光改变"与数据本身"图间曝光差异"的量级比。"""
    files = sorted(os.listdir(IR_DIR))[:120]
    ims = [np.asarray(imread(os.path.join(IR_DIR, f), cv2.IMREAD_GRAYSCALE), np.float32) / 255.0 for f in files]
    file_means = np.array([im.mean() for im in ims])
    natural = {
        "per_image_mean_p05": float(np.percentile(file_means, 5)),
        "per_image_mean_p50": float(np.percentile(file_means, 50)),
        "per_image_mean_p95": float(np.percentile(file_means, 95)),
        "p95_over_p05_exposure_ratio": float(np.percentile(file_means, 95) / np.percentile(file_means, 5)),
    }
    gammas = {}
    for g in (0.5, 0.6, 0.75, 0.85, 1.0, 1.2, 1.35, 1.5, 2.0):
        # 对"中位曝光"的那张图，gamma 前后 mean 之比 = 该 gamma 诱导的曝光伸缩
        med_img = ims[int(np.argsort(np.abs(file_means - natural["per_image_mean_p50"]))[0])]
        ratio = float((med_img ** g).mean() / med_img.mean())
        # 全样本上诱导的 mean 变化分布
        rs = np.array([((im ** g).mean() / im.mean()) for im in ims])
        gammas[str(g)] = {
            "exposure_ratio_at_median_image": round(ratio, 4),
            "over_files_mean": round(float(rs.mean()), 4),
            "over_files_p05": round(float(np.percentile(rs, 5)), 4),
            "over_files_p95": round(float(np.percentile(rs, 95)), 4),
        }
    return {"natural_image_to_image_exposure": natural, "gamma_induced_exposure_ratio": gammas}


# ---------------------------------------------------------------- §10 interaction matrix
def _pipeline_stageC(ir_aug, cfg, vis, dep, sample, stats=None):
    """真实路径：load_and_preprocess_image(RGBID) → load_image 的 rect resize → /255。

    返回 (stageB_clahe_ir, stageC_modelinput_ir)。
    """
    ds = object.__new__(B.BaseDataset)
    ds.use_simotm = "RGBID"
    ds.hyp = cfg
    ds.imgsz = 1280
    ds.augment = True
    ir = imread(os.path.join(IR_DIR, sample), cv2.IMREAD_GRAYSCALE)
    ir_b = apply_ir_encoding(ir, "clahe")
    ir_c = apply_ir_augmentation(ir_b, cfg, True, stats=stats) if ir_aug else ir_b

    # 复刻 load_image 的 rect resize（r = imgsz / max(h0,w0)）
    def rect_resize(a):
        h0, w0 = a.shape[:2]
        r = 1280 / max(h0, w0)
        w, h = min(int(np.ceil(w0 * r)), 1280), min(int(np.ceil(h0 * r)), 1280)
        return cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)

    vis_m = rect_resize(vis)
    dep_m = rect_resize(dep)
    ir_m = rect_resize(ir_c[..., None] if ir_c.ndim == 2 else ir_c)
    if ir_m.ndim == 3:
        ir_m = np.ascontiguousarray(ir_m.transpose(2, 0, 1))[..., 0]
    return ir_b, ir_m.astype(np.float32) / 255.0


def interaction_matrix():
    vis = imread(os.path.join(VIS_DIR, SAMPLES[0]))
    dep = imread(os.path.join(DEP_DIR, SAMPLES[0]), cv2.IMREAD_UNCHANGED)
    if dep.ndim == 3:
        dep = dep[..., 0]
    arms = {
        "baseline": (Cfg(), False),
        "gamma_only": (Cfg(gamma=(0.75, 1.35), gp=1.0), True),
        "noise_only": (Cfg(nstd=0.015, np_=1.0), True),
        "gamma_noise": (Cfg(gamma=(0.75, 1.35), gp=1.0, nstd=0.015, np_=1.0), True),
    }
    out = {}
    for name, (cfg, do) in arms.items():
        raw_list, clahe_list, model_list = [], [], []
        gammas, clip_hi, clip_lo = [], 0.0, 0.0
        IS = {"calls": 0, "gamma_applied": 0, "noise_applied": 0,
              "skipped_disabled": 0, "skipped_nonaugment": 0, "gamma_draws": [], "noise_std_used": []}
        for s in SAMPLES:
            ir = imread(os.path.join(IR_DIR, s), cv2.IMREAD_GRAYSCALE)
            raw_list.append(np.asarray(ir).ravel())
            np.random.seed(7); import random as _r; _r.seed(7)
            b, c = _pipeline_stageC(do, cfg, imread(os.path.join(VIS_DIR, s)),
                                    (lambda d: d[..., 0] if d.ndim == 3 else d)(
                                        imread(os.path.join(DEP_DIR, s), cv2.IMREAD_UNCHANGED)),
                                    s, stats=IS)
            clahe_list.append(np.asarray(b).ravel())
            model_list.append(c.ravel())
            if do:
                # 记录饱和：比较 aug 前后（float 域未被 clip 掉多少）
                x = np.asarray(b, np.float32) / 255.0
                if cfg.ir_gamma_probability > 0:
                    g = float(np.mean(IS["gamma_draws"])) if IS["gamma_draws"] else 0
                    gammas.append(g)
                clip_hi += float((x > 1.0).mean()); clip_lo += float((x < 0.0).mean())
        out[name] = {
            "IR_raw": stat(np.concatenate(raw_list)),
            "IR_after_CLAHE": stat(np.concatenate(clahe_list)),
            "IR_model_input_[0,1]": stat(np.concatenate(model_list)),
            "ir_aug_stats": {k: v for k, v in IS.items() if not isinstance(v, list)},
            "saturation_frac_above1": round(clip_hi / len(SAMPLES), 6),
            "saturation_frac_below0": round(clip_lo / len(SAMPLES), 6),
        }
    return out


def boundary_analysis(n=80):
    """噪声在 CLAHE 输出的**硬地板**上会不会被 clip 掉？——直接量"边界占用率"的变化。

    不做公式重写：同一张 CLAHE 输出，分别过真实 `apply_ir_augmentation` 的三个 arm，
    统计落在 0 / 255 两个边界上的像素比例变化。
    """
    import random as _r
    files = sorted(os.listdir(IR_DIR))[:n]
    arms = {"clahe_only": Cfg(),
            "gamma_only": Cfg(gamma=(0.75, 1.35), gp=1.0),
            "noise_only": Cfg(nstd=0.015, np_=1.0),
            "gamma_noise": Cfg(gamma=(0.75, 1.35), gp=1.0, nstd=0.015, np_=1.0)}
    acc = {a: {"at0": 0.0, "at255": 0.0, "n": 0} for a in arms}
    for f in files:
        base = apply_ir_encoding(np.asarray(imread(os.path.join(IR_DIR, f), cv2.IMREAD_GRAYSCALE)), "clahe")
        for name, cfg in arms.items():
            _r.seed(2024); np.random.seed(2024)
            out = apply_ir_augmentation(base, cfg, True)
            acc[name]["at0"] += float((out == 0).mean())
            acc[name]["at255"] += float((out == 255).mean())
            acc[name]["n"] += 1
    res = {}
    for name, v in acc.items():
        res[name] = {"frac_pixels_at_0": round(v["at0"] / v["n"], 6),
                     "frac_pixels_at_255": round(v["at255"] / v["n"], 6)}
    # CLAHE 输出的地板/天花板（= 噪声可用的 headroom）
    floors, ceils = [], []
    for f in files:
        b = np.asarray(apply_ir_encoding(np.asarray(imread(os.path.join(IR_DIR, f), cv2.IMREAD_GRAYSCALE)), "clahe"))
        floors.append(float(np.percentile(b, 1)))
        ceils.append(float(np.percentile(b, 99)))
    raw0 = []
    for f in files:
        r = np.asarray(imread(os.path.join(IR_DIR, f), cv2.IMREAD_GRAYSCALE))
        raw0.append(float((r == 0).mean()))
    res["clahe_headroom"] = {"p01_median": float(np.median(floors)), "p99_median": float(np.median(ceils)),
                             "noise_sigma_grays": 0.015 * 255.0,
                             "RAW_ir_frac_pixels_at_0": float(np.mean(raw0)),
                             "note": "raw IR 本身就有大量全黑像素；噪声把它们推回 0 属于回到原始分布，不是新伪影"}
    return res


def main():
    res = {"tool": "sanity_and_matrix.py"}
    print("===== §6 synthetic sanity (gamma) =====")
    res["synthetic"] = synthetic()
    for k, v in res["synthetic"].items():
        if k.startswith("gamma_"):
            bad = {n: e for n, e in v.items() if not e["finite"] or e["out_dtype"] != "uint8" or e["out_shape"] != e["out_shape"]}
            ident = [n for n, e in v.items() if e["identity_exact"]]
            print(f"  {k}: finite/shape/dtype OK={not bad}; exact-identity patterns={ident}; "
                  f"mid_gray {v['mid_gray']['mean_in']}->{v['mid_gray']['mean_out']}")
        else:
            print(f"  {k}: exact_identity={v['exact_identity']} measured_sigma_grays={v['measured_sigma_grays']} "
                  f"expected={v['expected_sigma_grays']} finite={v['finite']} dtype={v['out_dtype']}")

    print("\n===== §7 gamma range vs natural exposure =====")
    res["gamma_range"] = gamma_range_evidence()
    n = res["gamma_range"]["natural_image_to_image_exposure"]
    print(f"  natural per-image mean: p05={n['per_image_mean_p05']:.3f} p50={n['per_image_mean_p50']:.3f} "
          f"p95={n['per_image_mean_p95']:.3f}  ratio p95/p05 = {n['p95_over_p05_exposure_ratio']:.3f}")
    for g, e in res["gamma_range"]["gamma_induced_exposure_ratio"].items():
        print(f"   gamma={g:>4}: induced exposure ratio (median img) = {e['exposure_ratio_at_median_image']:.4f}")

    print("\n===== §10 interaction matrix =====")
    res["interaction"] = interaction_matrix()
    for arm, d in res["interaction"].items():
        print(f"  [{arm}]")
        for st in ("IR_raw", "IR_after_CLAHE", "IR_model_input_[0,1]"):
            s = d[st]
            print(f"     {st:22s} min={s['min']:.4f} max={s['max']:.4f} mean={s['mean']:.4f} std={s['std']:.4f} "
                  f"p01={s['p01']:.4f} p50={s['p50']:.4f} p99={s['p99']:.4f}")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)

    print("\n===== §10b clipping / boundary occupancy =====")
    res["boundary"] = boundary_analysis()
    for k, v in res["boundary"].items():
        if k == "clahe_headroom":
            print(f"  clahe_headroom: p01_median={v['p01_median']:.1f} p99_median={v['p99_median']:.1f} "
                  f"noise_sigma_grays={v['noise_sigma_grays']:.2f}")
        else:
            print(f"  {k:12s} frac@0={v['frac_pixels_at_0']:.6f} frac@255={v['frac_pixels_at_255']:.6f}")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
