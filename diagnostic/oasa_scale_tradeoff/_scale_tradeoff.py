"""_scale_tradeoff.py — OASA scale=1.4 vs 2.0 离线 geometry / supervision-retention 模拟。

性质：只读。0 次训练、0 次 A100、不修改 ObjectScaleAug 正式实现、不改任何正式配置。

口径声明（重要）：
  几何**不是**本脚本重新实现的 —— 它直接驱动 ultralytics/data/augment.py 里真实的
  `ObjectScaleAug` 算子。本脚本只负责：
    ① 用真实链路构建 dataset（build_yolo_dataset，与训练时完全相同的通路）
    ② 取出真实 labels（get_image_and_label 的返回，即 OASA 在训练中收到的同一个 dict）
    ③ 换掉算子的 scale，其余参数逐项固定
    ④ 统计该算子的 keep 决策造成的监督保留/丢失
  因此 retention 由**正式实现自己**算出，不是本脚本的近似。

  class 身份的传递：把 labels["cls"] 临时替换为 arange(n) 的索引标记，
  算子只把 cls 当不透明数组做 `cls[keep]`，从不读取其数值 → 幸存者索引可精确还原。
  真实类别在旁路数组 cls_true 中保留。这**不改变**任何几何/keep 判定。

paired 设计：每图固定 seed（SEED0+i）。算子内第一次随机调用是 prob 门、第二次是
  anchor 抽取，且 small 集合与 scale 无关 ⇒ 同一图在 1.4 / 2.0 下门控与 anchor 完全一致。

native 口径：small/medium/large 一律用 **native 像素面积**判定
  （small < 32^2 = 1024，medium < 96^2 = 9216，large >= 9216），与 STEP 2.5 / 4B 一致。
"""
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.augment import ObjectScaleAug  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.utils import yaml_load  # noqa: E402

OUT = Path(__file__).resolve().parent
DS_YAML = ROOT / "data/processed/rgbid_split_train/dataset.yaml"

SEED0 = 12345
SMALL_AREA = 1024.0
MEDIUM_AREA = 96.0 ** 2
SCALES = [("scale1.4", 1.4), ("scale2.0", 2.0)]
MOSAIC_FACTOR = 0.5  # Mosaic 四 tile：1280 -> 2560 画布再切回，最终输入空间 ×0.5


def bucket_of(a):
    if a < SMALL_AREA:
        return "small"
    if a < MEDIUM_AREA:
        return "medium"
    return "large"


def resol_group(h, w):
    """按 native 长边分组，便于与 STEP 2.5 的 1080p / 360p 对照。"""
    ls = max(h, w)
    if ls >= 1900:
        return "1080p"
    if ls <= 700:
        return "360p"
    return "other"


def main():
    data = yaml_load(str(DS_YAML))
    names = {int(k): v for k, v in data["names"].items()}
    overrides = dict(
        imgsz=1280, task="detect", rect=False, cache=False, single_cls=False,
        classes=None, fraction=1.0, channels=5, use_simotm="RGBID",
        mosaic=1.0, copy_paste=0.0, mixup=0.0,
        object_scale_aug=True, object_scale_aug_prob=0.5, object_scale_aug_scale=2.0,
        object_scale_aug_small_area=SMALL_AREA, object_scale_aug_min_visible=0.8,
        object_scale_aug_log_every=0,
    )
    cfg = get_cfg(overrides=overrides)
    img_path = (Path(data["path"]) / data["train"]).resolve()
    ds = build_yolo_dataset(
        cfg, str(img_path), 8, data, mode="train", use_simotm="RGBID",
        pairs_rgb_ir=["visible", "infrared"], pairs_rgb_depth=["visible", "depth"],
    )
    tmpl = ds.transforms.transforms[0]
    assert isinstance(tmpl, ObjectScaleAug), f"transforms[0] 不是 ObjectScaleAug: {type(tmpl)}"

    ops = {
        name: ObjectScaleAug(
            imgsz=tmpl.imgsz, enabled=True, prob=tmpl.prob, scale=s,
            small_area=tmpl.small_area, min_visible=tmpl.min_visible,
            mosaic_ref=tmpl.mosaic_ref, log_every=0,
        )
        for name, s in SCALES
    }
    print(f"[cfg] prob={tmpl.prob} small_area={tmpl.small_area} "
          f"min_visible={tmpl.min_visible} mosaic_p={tmpl.mosaic_runtime}")
    print(f"[cfg] scales = {[s for _, s in SCALES]}   seed0={SEED0}")

    n_img = len(ds)
    agg = {
        name: dict(
            gt_before=0, gt_after=0,
            b_before=Counter(), b_after=Counter(),
            c_before=Counter(), c_after=Counter(),
            r_before=Counter(), r_after=Counter(),
            stats_start=None,
            n_applied=0, n_img_loss=0, lost_per_img=[],
            cb_before=Counter(), cb_after=Counter(),
            anchor_before=[], anchor_after=[], anchor_vis=[],
            n_skip_nomosaic=0, n_skip_prob=0, n_skip_nosmall=0, n_skip_vis=0,
        )
        for name, _ in SCALES
    }
    base = dict(n_img_small=0, gt_total=0, small_total=0, resol=Counter(),
                shape_by_resol=defaultdict(Counter))
    op_stats = {name: ops[name].stats for name, _ in SCALES}
    # 逐 applied 图记录，用于「affected-image 内」口径（STEP 2.5 的 40.5%/58.7% 就是这个口径）
    applied_rec = {name: [] for name, _ in SCALES}

    for i in range(n_img):
        lab = ds.get_image_and_label(i)
        cls_true = np.asarray(lab["cls"]).astype(int).ravel()
        n = len(cls_true)
        if n == 0:
            continue
        H0, W0 = int(lab["ori_shape"][0]), int(lab["ori_shape"][1])
        bb = np.asarray(lab["instances"].bboxes)
        a_nat = np.clip(bb[:, 2] * W0, 0, None) * np.clip(bb[:, 3] * H0, 0, None)
        bk = np.array([bucket_of(x) for x in a_nat])
        rg = resol_group(H0, W0)
        base["gt_total"] += n
        base["small_total"] += int((a_nat < SMALL_AREA).sum())
        base["n_img_small"] += int(bool((a_nat < SMALL_AREA).any()))
        base["resol"][rg] += 1
        base["shape_by_resol"][rg][f"{H0}x{W0}"] += 1

        for name, op in ops.items():
            st = agg[name]
            st["gt_before"] += n
            for b in bk:
                st["b_before"][b] += 1
            for c in cls_true:
                st["c_before"][int(c)] += 1
            st["r_before"][rg] += n

            src = copy.deepcopy(lab)
            src["cls"] = np.arange(n, dtype=np.float32)
            n_anchor_before = len(op_stats[name]["anchor_before"])
            random.seed(SEED0 + i)
            op(src)
            applied = len(op_stats[name]["anchor_before"]) > n_anchor_before

            surv = np.asarray(src["cls"]).astype(int).ravel()
            ksurv = np.zeros(n, dtype=bool)
            ksurv[surv] = True
            st["gt_after"] += int(ksurv.sum())
            for j in np.where(ksurv)[0]:
                st["b_after"][bk[j]] += 1
                st["c_after"][int(cls_true[j])] += 1
            st["r_after"][rg] += int(ksurv.sum())

            if applied:
                st["n_applied"] += 1
                for c in cls_true:
                    st["cb_before"][int(c)] += 1
                for j in np.where(ksurv)[0]:
                    st["cb_after"][int(cls_true[j])] += 1
                lost = int((~ksurv).sum())
                st["lost_per_img"].append(lost)
                if lost > 0:
                    st["n_img_loss"] += 1
                s = op_stats[name]
                st["anchor_before"].append(s["anchor_before"][-1])
                st["anchor_after"].append(s["anchor_after"][-1])
                st["anchor_vis"].append(s["anchor_vis"][-1])
                applied_rec[name].append(dict(
                    resol=rg, buckets=bk.astype(str), survived=ksurv,
                    a_before=s["anchor_before"][-1], a_after=s["anchor_after"][-1],
                    a_vis=s["anchor_vis"][-1]))

        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{n_img}")

    for name, _ in SCALES:
        s = op_stats[name]
        agg[name].update(
            n_skip_nomosaic=s["skip_nomosaic"], n_skip_prob=s["skip_prob"],
            n_skip_nosmall=s["skip_nosmall"], n_skip_vis=s["skip_visibility"],
            n_eligible=s["eligible"], n_seen=s["seen"], n_anchor_kept=s["anchor_kept"],
        )

    def med(v):
        return float(np.median(v)) if len(v) else float("nan")

    summary = {
        "dataset": {
            "train_images": n_img,
            "total_GT": base["gt_total"],
            "native_small_GT": base["small_total"],
            "small_frac_of_GT": base["small_total"] / base["gt_total"],
            "small_containing_images": base["n_img_small"],
            "small_containing_frac": base["n_img_small"] / n_img,
            "native_shapes_by_resol": {k: dict(v) for k, v in base["shape_by_resol"].items()},
        },
        "operator_params": dict(prob=tmpl.prob, small_area=tmpl.small_area,
                                min_visible=tmpl.min_visible, seed0=SEED0,
                                mosaic_factor=MOSAIC_FACTOR),
        "conditions": {},
    }

    for name, s in SCALES:
        st = agg[name]
        gb, ga = st["gt_before"], st["gt_after"]
        cond = {
            "scale": s,
            "roi": {"width_ratio": 1.0 / s, "height_ratio": 1.0 / s,
                    "area_ratio": 1.0 / (s * s), "cropped_area_ratio": 1.0 - 1.0 / (s * s)},
            "gating": {"seen": st["n_seen"], "eligible": st["n_eligible"],
                       "applied": st["n_applied"],
                       "eligible_rate": st["n_eligible"] / st["n_seen"],
                       "applied_rate": st["n_applied"] / st["n_seen"],
                       "skip_nomosaic": st["n_skip_nomosaic"], "skip_prob": st["n_skip_prob"],
                       "skip_nosmall": st["n_skip_nosmall"], "skip_visibility": st["n_skip_vis"]},
            "anchor": {
                "n": len(st["anchor_before"]),
                "sqrt_area_before_input": med(st["anchor_before"]),
                "sqrt_area_after_input": med(st["anchor_after"]),
                "multiplier_input": med(st["anchor_after"]) / med(st["anchor_before"]),
                "sqrt_area_before_final": med(st["anchor_before"]) * MOSAIC_FACTOR,
                "sqrt_area_after_final": med(st["anchor_after"]) * MOSAIC_FACTOR,
                "multiplier_final": med(st["anchor_after"]) / med(st["anchor_before"]),
                "visibility_median": med(st["anchor_vis"]),
                "visibility_ge_0.8": float(np.mean(np.asarray(st["anchor_vis"]) >= 0.8))
                if st["anchor_vis"] else float("nan"),
                "anchor_kept": st["n_anchor_kept"],
                "anchor_kept_rate": st["n_anchor_kept"] / max(1, len(st["anchor_before"])),
            },
            "retention": {
                "all": {"before": gb, "after": ga, "retention": ga / gb, "loss": 1 - ga / gb},
            },
            "image_level": {
                "images_total": n_img,
                "images_with_small_GT": base["n_img_small"],
                "images_affected": st["n_applied"],
                "images_affected_frac": st["n_applied"] / n_img,
                "images_with_any_GT_lost": st["n_img_loss"],
                "images_with_any_GT_lost_frac": st["n_img_loss"] / n_img,
                "mean_GT_lost_per_affected": float(np.mean(st["lost_per_img"]))
                if st["lost_per_img"] else 0.0,
                "median_GT_lost_per_affected": med(st["lost_per_img"]),
            },
        }
        for b in ("small", "medium", "large"):
            bb_, ba_ = st["b_before"][b], st["b_after"][b]
            cond["retention"][b] = {
                "before": bb_, "after": ba_,
                "retention": (ba_ / bb_) if bb_ else float("nan"),
                "loss": (1 - ba_ / bb_) if bb_ else float("nan"),
            }
        cond["retention"]["small_of_dataset"] = st["b_after"]["small"] / st["b_before"]["small"] \
            if st["b_before"]["small"] else float("nan")

        # ---- affected-image 内口径：与 STEP 2.5 的 40.5% / 58.7% 同口径 ----
        recs = applied_rec[name]
        ai = dict(gt_before=0, gt_after=0, b_before=Counter(), b_after=Counter(),
                  r_before=Counter(), r_after=Counter(),
                  a_before=[], a_after=[], a_vis=[])
        for r in recs:
            nr = len(r["survived"])
            ai["gt_before"] += nr
            ai["gt_after"] += int(r["survived"].sum())
            for bb_, sv in zip(r["buckets"], r["survived"]):
                ai["b_before"][bb_] += 1
                ai["b_after"][bb_] += int(sv)
            ai["r_before"][r["resol"]] += nr
            ai["r_after"][r["resol"]] += int(r["survived"].sum())
            ai["a_before"].append(r["a_before"])
            ai["a_after"].append(r["a_after"])
            ai["a_vis"].append(r["a_vis"])
        gbb, gaa = ai["gt_before"], ai["gt_after"]
        cond["within_affected_image"] = {
            "n_images": len(recs), "GT_before": gbb, "GT_after": gaa,
            "retention": (gaa / gbb) if gbb else float("nan"),
            "drop": (1 - gaa / gbb) if gbb else float("nan"),
            "buckets": {
                b: dict(before=ai["b_before"][b], after=ai["b_after"][b],
                        retention=(ai["b_after"][b] / ai["b_before"][b]) if ai["b_before"][b] else float("nan"),
                        drop=(1 - ai["b_after"][b] / ai["b_before"][b]) if ai["b_before"][b] else float("nan"))
                for b in ("small", "medium", "large")},
            "by_resol": {rg: dict(before=ai["r_before"][rg], after=ai["r_after"][rg],
                                  drop=1 - ai["r_after"][rg] / ai["r_before"][rg])
                         for rg in ("1080p", "360p", "other") if ai["r_before"][rg]},
            "anchor_before_input": med(ai["a_before"]),
            "anchor_after_input": med(ai["a_after"]),
            "anchor_multiplier": med(ai["a_after"]) / med(ai["a_before"]),
            "anchor_vis_median": med(ai["a_vis"]),
            "anchor_vis_ge_0.8": float(np.mean(np.asarray(ai["a_vis"]) >= 0.8)),
        }
        # ---- 诊断：small 桶 drop 是否随「该图 small 数量」变化 ----
        # STEP 2.5 的 80 图平均 2.4 small/图（193/80），本次 209 图平均 3.8/图（789/209）。
        # 若 drop 随 small 密度上升，则两个数字的差异是**样本构成**而非口径。
        strat = {}
        for lo, hi, lab in ((1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 10**9, "6+")):
            sb = sa = imb = ima = 0
            for r in recs:
                k = int((r["buckets"] == "small").sum())
                if not (lo <= k <= hi):
                    continue
                m = r["buckets"] == "small"
                sb += int(m.sum())
                sa += int(r["survived"][m].sum())
                imb += int(r["survived"].sum() == len(r["survived"]))
                ima += 1
            if imb:
                strat[lab] = dict(images=ima, small_before=sb, small_after=sa,
                                  small_drop=(1 - sa / sb) if sb else float("nan"))
        cond["small_density_strata"] = strat

        # anchor 按分辨率分组（STEP 2.5 的 17.18 / 22.18 是分分辨率给的）
        cond["anchor_by_resol"] = {}
        for rg in ("1080p", "360p", "other"):
            sel = [r for r in recs if r["resol"] == rg]
            if sel:
                cond["anchor_by_resol"][rg] = dict(
                    n=len(sel),
                    sqrt_area_before_input=med([r["a_before"] for r in sel]),
                    sqrt_area_after_input=med([r["a_after"] for r in sel]),
                    multiplier=med([r["a_after"] for r in sel]) / med([r["a_before"] for r in sel]))
        cond["by_resol"] = {}
        for rg in ("1080p", "360p", "other"):
            rb, ra = st["r_before"][rg], st["r_after"][rg]
            if rb:
                cond["by_resol"][rg] = {"before": rb, "after": ra,
                                        "retention": ra / rb, "loss": 1 - ra / rb}
        cond["classes"] = {}
        for cid in sorted(names):
            cb, ca = st["c_before"][cid], st["c_after"][cid]
            ab, aa = st["cb_before"][cid], st["cb_after"][cid]
            cond["classes"][cid] = {
                "name": names[cid], "before": cb, "after": ca,
                "retention": (ca / cb) if cb else float("nan"),
                "loss": (1 - ca / cb) if cb else float("nan"),
                "affected_before": ab, "affected_after": aa,
                "affected_retention": (aa / ab) if ab else float("nan"),
                "affected_loss": (1 - aa / ab) if ab else float("nan"),
            }
        # 各桶「落在受影响图中的比例」—— dataset 层面换算的正确乘数（STEP 2.5 用的是 12.5% 图像占比）
        cond["affected_share_of_bucket"] = {
            b: (cond["within_affected_image"]["buckets"][b]["before"]
                / cond["retention"][b]["before"])
            for b in ("small", "medium", "large")
        }
        summary["conditions"][name] = cond

    # ---- trade-off / gain ----
    c14, c20 = summary["conditions"]["scale1.4"], summary["conditions"]["scale2.0"]
    summary["tradeoff"] = {
        "all_GT_retention_gain_1.4_over_2.0":
            c14["retention"]["all"]["retention"] - c20["retention"]["all"]["retention"],
        "small_retention_gain_1.4_over_2.0":
            c14["retention"]["small"]["retention"] - c20["retention"]["small"]["retention"],
        "medium_retention_gain_1.4_over_2.0":
            c14["retention"]["medium"]["retention"] - c20["retention"]["medium"]["retention"],
        "large_retention_gain_1.4_over_2.0":
            c14["retention"]["large"]["retention"] - c20["retention"]["large"]["retention"],
        "affected_images_gain_1.4_over_2.0":
            c14["image_level"]["images_affected"] - c20["image_level"]["images_affected"],
    }
    # dataset 层面（直接实测，非外推）
    for name in ("scale1.4", "scale2.0"):
        c = summary["conditions"][name]
        c["dataset_level_loss"] = {
            b: (1 - c["retention"][b]["after"] / c["retention"][b]["before"])
            * (c["retention"][b]["before"] / base["gt_total"])
            for b in ("small", "medium", "large")
        }

    (OUT / "_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[saved] {OUT / '_results.json'}")
    return summary


if __name__ == "__main__":
    main()
