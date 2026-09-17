"""scripts/rgbid_error_analysis.py — RGBID baseline 离线错误分析（只读）。

口径：验证集 400 图（rgbid_split/images/val/visible），预测来自
`diagnostic/rgbid_inference_gain/RGBID_baseline/results`（冻结 predict_rect.py 产出）。
匹配：逐类 + 按 confidence 降序贪心 + IoU>=0.5（与官方赛题口径的匹配规则一致）。

错误分类（预测框侧）：
  TP                  IoU>=0.5 且类别正确且该 GT 未被更高分框占用
  duplicate           同类别已有更高分框命中该 GT（重复预测）
  class_error         与「不同类别」GT 的 IoU>=0.5（类别错误）
  localization_error  与同类别 GT 的 0.1<=IoU<0.5（定位不准）
  background_fp       以上都不满足（纯背景误检）

漏检分类（GT 侧）：
  matched             被正确检出
  near_miss           有同类别预测但 0.1<=IoU<0.5（定位不足导致漏检）
  class_confusion     有不同类别预测且 IoU>=0.5（类别混淆导致漏检）
  hard_miss           完全无预测（纯漏检）

只读脚本：不修改任何冻结文件，不写 TXT，只输出统计与报告数据。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NC = 12
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
FOCUS = [0, 5, 7, 4, 1]  # person, bicycle, ball, sign, boat

LOOSE_IOU = 0.1     # 「接近命中」的下界
EPS = np.finfo(np.float32).eps


def read_txt(path: Path, with_conf: bool):
    n = 6 if with_conf else 5
    if not path.exists():
        return np.zeros((0, n), dtype=np.float32)
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s:
            continue
        p = s.split()
        if len(p) < n:
            continue
        rows.append([float(x) for x in p[:n]])
    return np.array(rows, dtype=np.float32).reshape(-1, n) if rows else np.zeros((0, n), np.float32)


def to_xyxy(a, w, h):
    if len(a) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    cx, cy, bw, bh = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    return np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h], axis=1), a[:, 0]


def iou_mat(a, b):
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), np.float32)
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="diagnostic/rgbid_inference_gain/RGBID_baseline/results")
    ap.add_argument("--split_root", default="data/processed/rgbid_split")
    ap.add_argument("--out_json", default="reports/rgbid_error_analysis.json")
    args = ap.parse_args()

    results_dir = (PROJECT_ROOT / args.results).resolve()
    split_root = (PROJECT_ROOT / args.split_root).resolve()
    images_dir = split_root / "images" / "val" / "visible"
    labels_dir = split_root / "labels" / "val" / "visible"

    # 每类统计容器
    st = {c: defaultdict(float) for c in range(NC)}
    conf_tp = {c: [] for c in range(NC)}
    conf_fp = {c: [] for c in range(NC)}
    gt_area = {c: [] for c in range(NC)}
    tp_area = {c: [] for c in range(NC)}
    fn_area = {c: [] for c in range(NC)}
    examples = defaultdict(list)          # 错误类型 -> [(stem, detail)]
    img_errors = defaultdict(int)         # stem -> 该图错误数（用于挑示例）

    n_images = 0
    n_corrupt = 0

    for img_path in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        stem = img_path.stem
        h, w = cv2.imread(str(img_path)).shape[:2]
        n_images += 1

        gt = read_txt(labels_dir / f"{stem}.txt", with_conf=False)
        if len(gt) and (gt[:, 1:].max() > 1.0 or gt[:, 1:].min() < 0.0):
            n_corrupt += 1
            continue
        gt_xyxy, gt_cls = to_xyxy(gt, w, h)
        gt_area_px = np.clip(gt_xyxy[:, 2] - gt_xyxy[:, 0], 0, None) * \
                     np.clip(gt_xyxy[:, 3] - gt_xyxy[:, 1], 0, None)
        img_area = float(w * h)

        pred = read_txt(results_dir / f"{stem}.txt", with_conf=True)
        pr_xyxy, pr_cls = to_xyxy(pred[:, :5], w, h) if len(pred) else (
            np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))
        pr_conf = pred[:, 5] if len(pred) else np.zeros((0,), np.float32)

        # GT 侧记录命中情况
        gt_matched = np.zeros(len(gt_xyxy), dtype=bool)
        gt_best_iou = np.zeros(len(gt_xyxy), dtype=np.float32)
        gt_best_same = np.zeros(len(gt_xyxy), dtype=np.float32)
        gt_best_diff = np.zeros(len(gt_xyxy), dtype=np.float32)

        # GT 计数
        for k in range(len(gt_xyxy)):
            c = int(gt_cls[k])
            st[c]["gt"] += 1
            gt_area[c].append(float(gt_area_px[k]))
            st[c]["gt_area_px_sum"] += float(gt_area_px[k])
            st[c]["gt_area_norm_sum"] += float(gt_area_px[k]) / img_area

        # 预测侧：逐类 conf 降序贪心
        order = np.argsort(-pr_conf) if len(pr_conf) else np.zeros(0, dtype=int)
        for i in order:
            c = int(pr_cls[i])
            st[c]["pred"] += 1
            st[c]["pred_conf_sum"] += float(pr_conf[i])
            if len(gt_xyxy) == 0:
                st[c]["background_fp"] += 1
                conf_fp[c].append(float(pr_conf[i]))
                examples[(c, "background_fp")].append(f"{stem}:{i}")
                continue
            ious = iou_mat(pr_xyxy[i:i + 1], gt_xyxy)[0]
            same = gt_cls == c
            # 更新该 GT 的最佳同/异类 IoU（GT 侧诊断用）
            if same.any():
                si = np.where(same)[0]
                bi = si[int(np.argmax(ious[si]))]
                gt_best_same[bi] = max(gt_best_same[bi], float(ious[bi]))
            if (~same).any():
                di = np.where(~same)[0]
                bi = di[int(np.argmax(ious[di]))]
                gt_best_diff[bi] = max(gt_best_diff[bi], float(ious[bi]))

            # 1) TP：同类未匹配 GT 且 IoU>=0.5
            cand = np.where(same & ~gt_matched & (ious >= 0.5))[0]
            if cand.size:
                j = cand[int(np.argmax(ious[cand]))]
                gt_matched[j] = True
                gt_best_iou[j] = float(ious[j])
                st[c]["tp"] += 1
                conf_tp[c].append(float(pr_conf[i]))
                tp_area[c].append(float(gt_area_px[j]))
                continue
            # 2) duplicate：同类已匹配 GT 且 IoU>=0.5
            if np.where(same & gt_matched & (ious >= 0.5))[0].size:
                st[c]["duplicate"] += 1
                conf_fp[c].append(float(pr_conf[i]))
                examples[(c, "duplicate")].append(f"{stem}:{i}")
                continue
            # 3) class_error：异类 GT 且 IoU>=0.5
            if np.where(~same & (ious >= 0.5))[0].size:
                st[c]["class_error"] += 1
                conf_fp[c].append(float(pr_conf[i]))
                examples[(c, "class_error")].append(f"{stem}:{i}")
                continue
            # 4) localization_error：同类 GT 且 0.1<=IoU<0.5
            if np.where(same & (ious >= LOOSE_IOU) & (ious < 0.5))[0].size:
                st[c]["localization_error"] += 1
                conf_fp[c].append(float(pr_conf[i]))
                examples[(c, "localization_error")].append(f"{stem}:{i}")
                continue
            # 5) background_fp
            st[c]["background_fp"] += 1
            conf_fp[c].append(float(pr_conf[i]))
            examples[(c, "background_fp")].append(f"{stem}:{i}")

        # 漏检分类
        for j in np.where(~gt_matched)[0] if len(gt_matched) else []:
            c = int(gt_cls[j])
            st[c]["fn"] += 1
            fn_area[c].append(float(gt_area_px[j]))
            if gt_best_diff[j] >= 0.5:
                st[c]["fn_class_confusion"] += 1
                examples[(c, "fn_class_confusion")].append(f"{stem}:gt{j}")
            elif gt_best_same[j] >= LOOSE_IOU:
                st[c]["fn_near_miss"] += 1
                examples[(c, "fn_near_miss")].append(f"{stem}:gt{j}")
            else:
                st[c]["fn_hard_miss"] += 1
                examples[(c, "fn_hard_miss")].append(f"{stem}:gt{j}")

        # 小/大目标（按 GT 归一化面积）
        for j in range(len(gt_xyxy)):
            c = int(gt_cls[j])
            a_norm = float(gt_area_px[j]) / img_area
            if a_norm < 0.001:
                st[c]["gt_small"] += 1
                if not gt_matched[j]:
                    st[c]["fn_small"] += 1
            elif a_norm > 0.05:
                st[c]["gt_large"] += 1
                if not gt_matched[j]:
                    st[c]["fn_large"] += 1

    # 汇总输出
    def pct(a, q):
        return float(np.percentile(a, q)) if len(a) else float("nan")

    out = {"n_images": n_images, "n_corrupt": n_corrupt, "per_class": {}}
    print("=" * 100)
    print("RGBID baseline 离线错误分析（val 400 图；IoU>=0.5 逐类 conf 贪心匹配）")
    print("=" * 100)
    print(f"图像 {n_images}（跳过 corrupt {n_corrupt}）")
    print()
    hdr = (f"{'class':<12}{'GT':>5}{'pred':>6}{'TP':>6}{'FN':>5}{'FP':>6}"
           f"{'loc_err':>8}{'cls_err':>8}{'dup':>5}{'bg_fp':>6}")
    print(hdr)
    print("-" * len(hdr))
    tot = defaultdict(float)
    for c in range(NC):
        s = st[c]
        fp = s["duplicate"] + s["class_error"] + s["localization_error"] + s["background_fp"]
        print(f"{NAMES[c]:<12}{int(s['gt']):>5}{int(s['pred']):>6}{int(s['tp']):>6}"
              f"{int(s['fn']):>5}{int(fp):>6}{int(s['localization_error']):>8}"
              f"{int(s['class_error']):>8}{int(s['duplicate']):>5}{int(s['background_fp']):>6}")
        for k in ("gt", "pred", "tp", "fn", "duplicate", "class_error",
                  "localization_error", "background_fp"):
            tot[k] += s[k]
    print("-" * len(hdr))
    print(f"{'ALL':<12}{int(tot['gt']):>5}{int(tot['pred']):>6}{int(tot['tp']):>6}"
          f"{int(tot['fn']):>5}{int(tot['duplicate']+tot['class_error']+tot['localization_error']+tot['background_fp']):>6}"
          f"{int(tot['localization_error']):>8}{int(tot['class_error']):>8}"
          f"{int(tot['duplicate']):>5}{int(tot['background_fp']):>6}")

    print()
    print("漏检细分（FN 的成因）:")
    h2 = f"{'class':<12}{'FN':>6}{'near_miss':>11}{'cls_conf':>10}{'hard_miss':>11}"
    print(h2); print("-" * len(h2))
    for c in range(NC):
        s = st[c]
        if s["gt"] == 0:
            continue
        print(f"{NAMES[c]:<12}{int(s['fn']):>6}{int(s['fn_near_miss']):>11}"
              f"{int(s['fn_class_confusion']):>10}{int(s['fn_hard_miss']):>11}")

    print()
    print("置信度分布（TP vs FP）:")
    h3 = (f"{'class':<12}{'TP_conf_mean':>13}{'FP_conf_mean':>13}"
          f"{'TP_p10':>9}{'TP_p90':>9}{'FP_p10':>9}{'FP_p90':>9}")
    print(h3); print("-" * len(h3))
    for c in range(NC):
        if st[c]["gt"] == 0:
            continue
        t, f = conf_tp[c], conf_fp[c]
        print(f"{NAMES[c]:<12}{np.mean(t) if t else float('nan'):>13.4f}"
              f"{np.mean(f) if f else float('nan'):>13.4f}"
              f"{pct(t,10):>9.4f}{pct(t,90):>9.4f}{pct(f,10):>9.4f}{pct(f,90):>9.4f}")

    print()
    print("目标面积分布（GT 归一化面积；small<0.1% / large>5%）:")
    h4 = (f"{'class':<12}{'gt':>6}{'gt_small':>10}{'fn_small':>10}"
          f"{'gt_large':>10}{'fn_large':>10}{'area_med':>11}")
    print(h4); print("-" * len(h4))
    for c in range(NC):
        s = st[c]
        if s["gt"] == 0:
            continue
        med = float(np.median(gt_area[c])) if gt_area[c] else float("nan")
        print(f"{NAMES[c]:<12}{int(s['gt']):>6}{int(s['gt_small']):>10}{int(s['fn_small']):>10}"
              f"{int(s['gt_large']):>10}{int(s['fn_large']):>10}{med:>11.0f}")

    # 重点类详表
    print()
    print("=" * 100)
    print("重点类别详表")
    print("=" * 100)
    for c in FOCUS:
        s = st[c]
        fp = s["duplicate"] + s["class_error"] + s["localization_error"] + s["background_fp"]
        recall = s["tp"] / s["gt"] if s["gt"] else float("nan")
        prec = s["tp"] / (s["tp"] + fp) if (s["tp"] + fp) else float("nan")
        print(f"\n{NAMES[c]}  (class {c})")
        print(f"  GT={int(s['gt'])}  pred={int(s['pred'])}  TP={int(s['tp'])}  "
              f"FN={int(s['fn'])}  FP={int(fp)}")
        print(f"  recall={recall:.4f}  precision={prec:.4f}")
        print(f"  FP 细分: 定位错误={int(s['localization_error'])}  类别错误={int(s['class_error'])}  "
              f"重复={int(s['duplicate'])}  纯背景={int(s['background_fp'])}")
        print(f"  FN 细分: 定位不足={int(s['fn_near_miss'])}  类别混淆={int(s['fn_class_confusion'])}  "
              f"纯漏检={int(s['fn_hard_miss'])}")
        t, f = conf_tp[c], conf_fp[c]
        print(f"  conf: TP均值={np.mean(t) if t else float('nan'):.4f}  "
              f"FP均值={np.mean(f) if f else float('nan'):.4f}  "
              f"TP_p10={pct(t,10):.4f}  FP_p90={pct(f,90):.4f}")
        print(f"  面积(px) 中位数: GT={np.median(gt_area[c]) if gt_area[c] else float('nan'):.0f}  "
              f"TP={np.median(tp_area[c]) if tp_area[c] else float('nan'):.0f}  "
              f"FN={np.median(fn_area[c]) if fn_area[c] else float('nan'):.0f}")
        for k in ("background_fp", "localization_error", "fn_near_miss", "fn_hard_miss", "class_error"):
            ex = [e.split(":")[0] for e in examples[(c, k)]][:5]
            print(f"  示例图片({k}): {ex}")

    # JSON 存档
    for c in range(NC):
        s = st[c]
        out["per_class"][NAMES[c]] = {
            "class_id": c,
            "gt": int(s["gt"]), "pred": int(s["pred"]), "tp": int(s["tp"]), "fn": int(s["fn"]),
            "fp": int(s["duplicate"] + s["class_error"] + s["localization_error"] + s["background_fp"]),
            "localization_error": int(s["localization_error"]),
            "class_error": int(s["class_error"]),
            "duplicate": int(s["duplicate"]),
            "background_fp": int(s["background_fp"]),
            "fn_near_miss": int(s["fn_near_miss"]),
            "fn_class_confusion": int(s["fn_class_confusion"]),
            "fn_hard_miss": int(s["fn_hard_miss"]),
            "gt_small": int(s["gt_small"]), "fn_small": int(s["fn_small"]),
            "gt_large": int(s["gt_large"]), "fn_large": int(s["fn_large"]),
            "tp_conf_mean": float(np.mean(conf_tp[c])) if conf_tp[c] else None,
            "fp_conf_mean": float(np.mean(conf_fp[c])) if conf_fp[c] else None,
            "gt_area_px_median": float(np.median(gt_area[c])) if gt_area[c] else None,
            "tp_area_px_median": float(np.median(tp_area[c])) if tp_area[c] else None,
            "fn_area_px_median": float(np.median(fn_area[c])) if fn_area[c] else None,
        }
    totfp = int(tot["duplicate"] + tot["class_error"] + tot["localization_error"] + tot["background_fp"])
    out["overall"] = {
        "gt": int(tot["gt"]), "pred": int(tot["pred"]), "tp": int(tot["tp"]),
        "fn": int(tot["fn"]), "fp": totfp,
        "localization_error": int(tot["localization_error"]),
        "class_error": int(tot["class_error"]),
        "duplicate": int(tot["duplicate"]),
        "background_fp": int(tot["background_fp"]),
    }
    outp = (PROJECT_ROOT / args.out_json).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] -> {outp}")


if __name__ == "__main__":
    main()
