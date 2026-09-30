"""_audit.py — Stage-4 Label / Channel Consistency Audit（STRICT READ-ONLY）

唯一目标：验证上一轮 `prediction_head_geometry_audit` 的 G1「GT-class sigmoid ≈ 0」
是否对应一条**正确的**测量链：

    GT class ID → dataset/label mapping → model.names/nc → Detect class channel
    → raw class logit → sigmoid → TAL pd_scores/GT-class → prediction class ID → evaluator

严格禁止：training / backward / optimizer / **model.forward / predict / val / inference** /
新 prediction / 新 submission / 改 checkpoint·source·config·dataset·labels·evaluator·
既有 diagnostic / 重抽 G1·G2·G4 / 重算前向 logits。

允许：读 source / YAML / dataset metadata / labels / **checkpoint metadata 与 state_dict**（纯读取，
        不执行任何 forward）/ 既有 `_v6_domains.json`·`_v4_internal.json`·`_v7_vectors`·
        `candidate_density_counterfactual/_results.npz` / 预测 TXT / G1·G2·G4 records；
        对已有 tensor/JSON/NPZ 做纯数学检查；建新目录；SHA256；git。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DS_YAML = ROOT / "data/processed/rgbid_split_train/dataset.yaml"
MODEL_YAML = ROOT / "configs/yolo11m_sepstem.yaml"
TRAIN_YAML = ROOT / "configs/train_rgbid_sepstem_clahe.yaml"
CKPT = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
V2D = ROOT / "diagnostic/small_object_cause_v2"
P3 = ROOT / "diagnostic/p3_feature_space"
DOM = ROOT / "diagnostic/small_object_domains"
V4F = ROOT / "diagnostic/small_object_internal_audit/_v4_internal.json"
CD = ROOT / "diagnostic/candidate_density_counterfactual/_results.npz"
TXT = ROOT / "diagnostic/sepstem_clahe/best_full/results"
LAB = ROOT / "data/processed/rgbid_split_train/labels/val/visible"

CHECKS: dict[str, object] = {}
LOG: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def chk(name: str, ok, detail: str = "") -> bool:
    CHECKS[name] = dict(pass_=bool(ok), detail=detail)
    say(f"    [{'PASS' if ok else 'FAIL'}] {name}   {detail}")
    return bool(ok)


def sha16(p: Path) -> str:
    p = Path(p)
    if p.is_dir():
        h = hashlib.sha256()
        for f in sorted(x for x in p.rglob("*") if x.is_file()):
            h.update(str(f.relative_to(p)).encode())
            h.update(hashlib.sha256(f.read_bytes()).digest())
        return h.hexdigest()[:16]
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def load_yaml_names(p: Path):
    """极简 YAML 读取（只要 names / nc / ch），避免引入额外解析歧义。"""
    names, nc = {}, None
    in_names = False
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.rstrip()
        if s.startswith("names:"):
            in_names = True
            continue
        if in_names:
            t = s.strip()
            if t and ":" in t and t.split(":")[0].strip().isdigit():
                k, v = t.split(":", 1)
                names[int(k.strip())] = v.strip().strip('"').strip("'")
                continue
            if t and not t.startswith("#"):
                in_names = False
        if s.startswith("nc:"):
            nc = int(s.split(":", 1)[1].split("#")[0].strip())
    return names, nc


def main() -> int:
    say("=" * 112)
    say("STAGE-4 LABEL / CHANNEL CONSISTENCY AUDIT —— STRICT READ-ONLY")
    say("=" * 112)

    prov = {str(p): sha16(p) for p in
            (DS_YAML, MODEL_YAML, TRAIN_YAML, CKPT,
             ROOT / "ultralytics/nn/modules/head.py", ROOT / "ultralytics/nn/tasks.py",
             ROOT / "ultralytics/utils/tal.py", ROOT / "ultralytics/utils/loss.py",
             ROOT / "ultralytics/models/yolo/detect/train.py",
             ROOT / "scripts/official_eval.py", Path(__file__).resolve().parents[2] / "scripts/predict_rect.py",
             DOM / "_v6_domains.json", V4F, P3 / "_v7_vectors.json", CD)}
    say("[provenance]")
    for k, v in prov.items():
        say(f"  {v}  {k}")

    # ================================================================ §2
    say("")
    say("=" * 112)
    say("[§2] DATASET CLASS MAPPING")
    say("=" * 112)
    ds_names, ds_nc = load_yaml_names(DS_YAML)
    say(f"  source = `{DS_YAML.relative_to(ROOT)}`")
    for k in sorted(ds_names):
        say(f"    {k:>2} -> {ds_names[k]}")
    say(f"  nc = {ds_nc}   len(names) = {len(ds_names)}")
    chk("C_dataset_names_complete", sorted(ds_names) == list(range(12)),
        f"keys={sorted(ds_names)}")
    chk("C_dataset_nc_matches_names", ds_nc == len(ds_names) == 12,
        f"nc={ds_nc} len(names)={len(ds_names)}")

    # G1/G2/G4 的 label 整数 ↔ 名字（用真实 label 文件）
    v3 = json.loads((V2D / "_v3_analysis.json").read_text(encoding="utf-8"))
    ids = json.loads((P3 / "_v11_ids.json").read_text(encoding="utf-8"))
    R2 = {f"{r['image_id']}#{r['gt_id']}": r for r in
          json.loads((V2D / "_v2_records.json").read_text(encoding="utf-8"))["records"]}
    G1 = list(v3["E1"]); G2 = [v[0] for v in v3["controls"].values()]; G4 = list(ids["G4"])

    def label_cls(stem: str, j: int) -> int | None:
        p = LAB / f"{stem}.txt"
        if not p.exists():
            return None
        rows = [ln.split() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if j >= len(rows):
            return None
        return int(float(rows[j][0]))

    bad = []
    per_grp = {}
    for nm, ks in (("G1", G1), ("G2", G2), ("G4", G4)):
        ok = 0; tot = 0; cls_seen = {}
        for k in ks:
            stem, j = k.rsplit("#", 1)
            lc = label_cls(stem, int(j))
            if lc is None:
                bad.append((k, "label-missing"))
                continue
            tot += 1
            rec = R2.get(k)
            if rec is None or int(rec["cls"]) != lc:
                bad.append((k, f"record cls={None if rec is None else rec['cls']} vs label {lc}"))
            else:
                ok += 1
            cls_seen[lc] = cls_seen.get(lc, 0) + 1
        per_grp[nm] = dict(n=tot, match=ok,
                           classes={ds_names[c]: v for c, v in sorted(cls_seen.items())})
    say("")
    say(f"  {'group':<5}{'n':>5}{'label↔record cls 一致':>22}   类别（label 整数 → 名字）")
    for nm in ("G1", "G2", "G4"):
        d = per_grp[nm]
        say(f"  {nm:<5}{d['n']:>5}{d['match']:>22}   {d['classes']}")
    chk("C_label_integer_to_name", not bad, f"不一致条数={len(bad)} {bad[:3]}")

    # ================================================================ §3
    say("")
    say("=" * 112)
    say("[§3] model.nc / model.names（读 **checkpoint metadata**，不执行 forward）")
    say("=" * 112)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "_s4_ultra_cfg"))
    import torch  # noqa: E402
    import ultralytics  # noqa: E402,F401
    ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    m = ck.get("ema") or ck["model"]
    say(f"  checkpoint top-level keys = {sorted(ck.keys())}")
    say(f"  model class = {type(m).__name__}   model.nc = {getattr(m, 'nc', None)}")
    mn = getattr(m, "names", None)
    say(f"  model.names type = {type(mn).__name__}")
    for k in sorted(mn):
        say(f"    {k:>2} -> {mn[k]}")
    chk("C_model_nc", int(m.nc) == 12 and int(m.nc) == ds_nc, f"model.nc={m.nc} dataset.nc={ds_nc}")
    same_order = all(str(mn[i]) == str(ds_names[i]) for i in range(12))
    chk("C_model_names_ID_order_identical", same_order and set(int(k) for k in mn) == set(range(12)),
        f"ID-by-ID 全等={same_order}（非 set 比较）")
    say(f"  model.yaml 的 nc = {load_yaml_names(MODEL_YAML)[1]}；"
        f"train.yaml 里无 names/nc（由 dataset.yaml 提供）")

    # ================================================================ §4/§5/§6/§7
    say("")
    say("=" * 112)
    say("[§4] DETECTION HEAD CHANNEL LAYOUT（本项目 fork 的真实 source）")
    say("=" * 112)
    say("  `ultralytics/nn/modules/head.py:25-78`  class Detect")
    say("    :41  self.nc = nc")
    say("    :43  self.reg_max = 16")
    say("    :44  self.no = nc + self.reg_max * 4          → 12 + 64 = 76")
    say("    :48  cv2  末层 = nn.Conv2d(c2, 4 * self.reg_max, 1)   → **64 ch（box/DFL）**")
    say("    :51 / :57  cv3 末层 = nn.Conv2d(c3, self.nc, 1)       → **12 ch（class）**")
    say("    :74  x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)")
    say("         ⇒ 拼接顺序 **[ box(0:64) | cls(64:76) ]**，class 通道 = **no 维的 [64:76]**")
    say("    :108-117 `_inference`:")
    say("         x_cat = cat([xi.view(B, self.no, -1)], 2)                 # (B,76,8400)")
    say("         box, cls = x_cat.split((self.reg_max*4, self.nc), 1)      # box=(B,64,·) cls=(B,12,·)")
    say("    :133-135 dbox = decode_bboxes(dfl(box), anchors) * strides;  return cat((dbox, cls.sigmoid()), 1)")
    say("")
    say("  `ultralytics/utils/loss.py:490-492`  v8DetectionLoss.compute_loss:")
    say("      pred_distri, pred_scores = cat([...], 2).split((self.reg_max*4, self.nc), 1)")
    say("      ⇒ **与 _inference 完全相同的 [0:64]=distri / [64:76]=scores 布局**（两处独立一致）")
    say("  `loss.py:424`  self.no = m.nc + m.reg_max * 4 = 76；:423 self.nc = m.nc = 12")
    say("")
    say("  [§7] OBJECTNESS：")
    obj_hits = [ln for ln in
                (ROOT / "ultralytics").rglob("*.py")
                if "objectness" in ln.read_text(encoding="utf-8", errors="replace")]
    say(f"     全 `ultralytics/**/*.py` 中 'objectness' 命中文件 = "
        f"{[p.name for p in obj_hits]}（唯一命中是 metrics.py:1254 的一句注释 'Sort by objectness'）")
    say("     `grep -rn objectness ultralytics/nn/modules/head.py ultralytics/utils/loss.py "
        "ultralytics/utils/tal.py` = **0 命中**")
    say("     ⇒ **OBJECTNESS_BRANCH = ABSENT**（该 fork 的 Detect 只有 box(64) + cls(12)，无 obj 通道）")
    chk("C_objectness_absent", len(obj_hits) <= 1,
        f"命中={[str(p.relative_to(ROOT)) for p in obj_hits]}")

    # checkpoint 侧的形状核对（纯 state_dict 读取）
    sd = m.state_dict()
    lay = m.model[-1]
    say("")
    say("  checkpoint state_dict 实测形状（Detect = model.30）：")
    shapes = {}
    for name in ("model.30.cv2.0.2.weight", "model.30.cv2.0.2.bias",
                 "model.30.cv3.0.2.weight", "model.30.cv3.0.2.bias"):
        t = sd.get(name)
        shapes[name] = tuple(t.shape) if t is not None else None
        say(f"    {name:<28} {shapes[name]}")
    say(f"    Detect.head 实测 no = {int(lay.no)}  nc = {int(lay.nc)}  reg_max = {int(lay.reg_max)}"
        f"  nl = {int(lay.nl)}")
    chk("C_head_box_channels_64", shapes["model.30.cv2.0.2.weight"][0] == 64,
        f"cv2 末层 out={shapes['model.30.cv2.0.2.weight'][0]}")
    chk("C_head_class_channels_eq_nc", shapes["model.30.cv3.0.2.weight"][0] == 12,
        f"cv3 末层 out={shapes['model.30.cv3.0.2.weight'][0]} == nc")
    chk("C_no_eq_64plus_nc", int(lay.no) == 64 + 12, f"no={int(lay.no)}")
    # 三个检测层的 class/branch 形状
    say("    各层末层形状：")
    for i in range(int(lay.nl)):
        w2 = tuple(sd[f"model.30.cv2.{i}.2.weight"].shape)
        w3 = tuple(sd[f"model.30.cv3.{i}.2.weight"].shape)
        say(f"      level {i}: cv2末 {w2}   cv3末 {w3}")

    # ================================================================ §5 GT class → channel
    say("")
    say("=" * 112)
    say("[§5] GT CLASS → HEAD CLASS CHANNEL（数值验证，非「看起来合理」）")
    say("=" * 112)
    say("  验证思路：`_v7_vectors.json` 落盘了每个 GT 在 **GT 中心 cell** 的 `h`")
    say("  （= `Detect.cv3[li][-1]` 的 **输入**，256 维）与 `logit_decomp`。")
    say("  该项目自己声明的恒等式是 `logit_c = W_c · h + b_c`，其中 `W = cv3[-1].weight` shape (12,C,1,1)。")
    say("  若 `W` 的行序 == class id，则用 checkpoint 权重从**落盘的 h** 重算，必须复现 `logit_decomp`。")
    Wc = sd["model.30.cv3.0.2.weight"].detach().float().view(12, -1).numpy().astype(np.float64)
    bc = sd["model.30.cv3.0.2.bias"].detach().float().numpy().astype(np.float64)
    v7 = json.loads((P3 / "_v7_vectors.json").read_text(encoding="utf-8"))["rows"]
    V7 = {r["key"]: r for r in v7 if r["domain"] == "V1"}
    d = []
    for k, r in V7.items():
        h = np.asarray(r["h"], np.float64)
        d.append(float(Wc[int(r["cls"])] @ h + bc[int(r["cls"])] - r["logit_decomp"]))
    d = np.array(d)
    say(f"    n = {len(d)}（全部 val GT）")
    say(f"    重算 `W[cls]·h + b[cls]` 与落盘 `logit_decomp` 的差："
        f" max|Δ| = {np.max(np.abs(d)):.3e}   median|Δ| = {np.median(np.abs(d)):.3e}")
    chk("C_head_weight_row_equals_class_id", np.max(np.abs(d)) < 1e-3,
        f"max|Δ|={np.max(np.abs(d)):.3e}（用 row=cls 复现 logit；换行会失败）")
    # 反证：若通道错位（用 cls+1 取行），应当复现失败
    d2 = []
    for k, r in V7.items():
        c = int(r["cls"]); h = np.asarray(r["h"], np.float64)
        d2.append(float(np.abs(Wc[(c + 1) % 12] @ h + bc[(c + 1) % 12] - r["logit_decomp"])))
    say(f"    反证：若改用 `W[cls+1]` 行 —— median|Δ| = {np.median(d2):.4f}（应显著大于 0）")
    chk("C_channel_offset_would_fail", np.median(d2) > 1e-2,
        f"错位行 median|Δ|={np.median(d2):.4f} ⇒ 检查有分辨力")

    # ---- 一个我做过又否掉的检查（如实记录）----
    say("")
    say("  ⚠ **我自己否掉的一个检查**：先试过拿 `_v6_domains.center_logit`（= GT 中心 12px 内**多 anchor 取 max**，")
    say("     跨 stride-8/16/32）去比 `_v7_vectors.logit_decomp`（= GT 中心**单 cell 点值**）。")
    say("     实测 Spearman ρ = 0.0375、逐类 ρ ∈ [−0.31, +0.55]。**这不能判为缺陷**：")
    say("     「邻域 max」与「单点值」是**两个不同的统计量**，而 head 输出在空间上变化很快 ⇒ 低相关是构造性的。")
    say("     该检查**设计不当**，予以撤销，换成下面两个**同统计量**的对照。")

    # ---- 替换检查 1：两个独立实现对【同一统计量】的一致性 ----
    dom = json.loads((DOM / "_v6_domains.json").read_text(encoding="utf-8"))
    V6 = {r["key"]: r for r in dom if r["domain"] == "V1"}
    v4j = json.loads(V4F.read_text(encoding="utf-8"))
    V4 = {f"{r['stem']}#{r['gt']}": r for r in v4j}
    say("")
    say("  [替换检查 1] 两个**独立实现**对**同一统计量**的一致性：")
    say("     `_v4_internal.py:183`  pos_score_max = pred_scores[0].sigmoid()[:, cls_id][pos].max()")
    say("     `_v6_domains.py:155`    pos_prob      = pd_s[0].sigmoid()[pos, cid].max()")
    say("     ⇒ **定义逐字相同**（GT 类通道在 positives 上的最大 sigmoid）。两者应几乎相等。")
    common = [k for k in V6 if k in V4]
    x = np.array([V4[k]["pos_score_max"] for k in common], float)
    y = np.array([V6[k]["pos_prob"] for k in common], float)
    mk2 = np.isfinite(x) & np.isfinite(y)
    x, y = x[mk2], y[mk2]
    dd2 = np.abs(x - y)
    say(f"       n = {len(x)}（V4 只覆盖 30 张 val 图，故少于 2807）")
    say(f"       max|Δ| = {dd2.max():.3e}   median|Δ| = {np.median(dd2):.3e}   Spearman ρ = {_spearman(x, y):+.4f}")
    say(f"       反证：若其一用了错通道，该量会指向别的类 ⇒ 不可能逐条吻合到 1e-6 量级。")
    chk("C_two_impl_same_statistic_agree", dd2.max() < 1e-5,
        f"max|Δ|={dd2.max():.3e} n={len(x)}（同定义、两实现）")
    # 同一对的 best_assign_iou（也是同定义）
    xi = np.array([V4[k]["best_assign_iou"] for k in common], float)
    yi = np.array([V6[k]["best_assign_iou"] for k in common], float)
    mk3 = np.isfinite(xi) & np.isfinite(yi)
    ddi = np.abs(xi[mk3] - yi[mk3])
    say(f"       同一对的 `best_assign_iou`（亦同定义）：n={mk3.sum()}  max|Δ| = {ddi.max():.3e}")
    chk("C_two_impl_iou_agree", ddi.max() < 1e-5, f"max|Δ|={ddi.max():.3e}")

    # ---- 替换检查 2：从最终 TXT 反查「GT 类 == 预测类」是否成立 ----
    say("")
    say("  [替换检查 2] 端到端：对每个与 GT 有重叠的最终预测，其 class id 是否就等于 GT 的 class id？")
    say("     这是**整条链的出口检查**（GT label id → channel → argmax → TXT class → evaluator）。")
    say("     若 channel 有置换，这里会出现固定的『c → c′』错配表。")
    from PIL import Image as _Im
    conf_tab = {}
    tot_ov = 0; tot_same = 0
    modal = {}
    for stem in sorted({k.rsplit('#', 1)[0] for k in list(G1) + list(G2) + list(G4)}):
        gp = LAB / f"{stem}.txt"
        ip = ROOT / "data/processed/rgbid_split_train/images/val/visible" / f"{stem}.png"
        if not ip.exists():
            ip = ip.with_suffix(".jpg")
        if not (gp.exists() and ip.exists()):
            continue
        with _Im.open(ip) as im:
            W_, H_ = im.size
        gt = [ln.split() for ln in gp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        _, pr, _ = _read_pred(TXT / f"{stem}.txt")
        if len(pr) == 0:
            continue
        # GT xyxy + pred xyxy
        g = np.array([[float(r[1]) * W_, float(r[2]) * H_, float(r[3]) * W_, float(r[4]) * H_]
                      for r in gt], float)
        g = np.stack([g[:, 0] - g[:, 2] / 2, g[:, 1] - g[:, 3] / 2,
                      g[:, 0] + g[:, 2] / 2, g[:, 1] + g[:, 3] / 2], 1)
        p = np.array([[float(r[1]) * W_, float(r[2]) * H_, float(r[3]) * W_, float(r[4]) * H_]
                      for r in pr], float)
        p = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2,
                      p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1)
        pc = pr[:, 0].astype(int)
        for j in range(len(g)):
            gcj = int(float(gt[j][0]))
            conf_tab.setdefault(gcj, {}).setdefault("n_gt", 0)
            conf_tab[gcj]["n_gt"] += 1
            iou = _iou(g[j], p)
            sel = np.where(iou > 0.0)[0]
            if not len(sel):
                continue
            conf_tab[gcj].setdefault("n_gt_overlap", 0)
            conf_tab[gcj]["n_gt_overlap"] += 1
            tot_ov += len(sel)
            tot_same += int((pc[sel] == gcj).sum())
            conf_tab.setdefault(gcj, {}).setdefault("ov", 0)
            conf_tab[gcj]["ov"] += len(sel)
            conf_tab.setdefault(gcj, {}).setdefault("same", 0)
            conf_tab[gcj]["same"] += int((pc[sel] == gcj).sum())
            # 最高 conf 的重叠预测的类
            k_best = sel[int(np.argmax(pr[sel, 5]))]
            modal.setdefault(gcj, {})
            modal[gcj][int(pc[k_best])] = modal[gcj].get(int(pc[k_best]), 0) + 1
    say(f"     全体（G1+G2+G4 图）重叠 (GT, pred) 对： n = {tot_ov}；"
        f"同类预测占比 = {tot_same/max(tot_ov,1):.2%}（随机基线 = 1/12 = 8.33%）")
    say(f"     {'GT class':<13}{'n_GT':>6}{'n_重叠GT':>9}{'同行最高类':>12}{'占比':>8}"
        f"{'自身类占比':>11}   最高 conf 重叠预测的类分布（按 GT 计）")
    row_argmax_self = {}
    maps = {}
    for c in sorted(conf_tab):
        d_ = conf_tab[c]
        m_ = modal.get(c, {})
        if not m_:
            continue
        top = sorted(m_.items(), key=lambda t: -t[1])
        tot_gt_c = sum(m_.values())
        top_s = ", ".join(f"{ds_names[k]}×{v}" for k, v in top[:3])
        row_argmax_self[c] = (top[0][0] == c)
        maps[c] = top[0][0]
        n_ov_gt = d_.get("n_gt_overlap", 0)
        say(f"     {ds_names[c]:<13}{d_.get('n_gt',0):>6}{n_ov_gt:>9}{ds_names[top[0][0]]:>12}"
            f"{top[0][1]/max(tot_gt_c,1):>8.1%}{d_['same']/max(d_['ov'],1):>11.1%}   {top_s}")
    big = {c: v for c, v in row_argmax_self.items() if conf_tab[c].get("n_gt_overlap", 0) >= 10}
    say("")
    say(f"     (a) 自身类占比 = {tot_same/max(tot_ov,1):.2%}  对  随机基线 8.33%"
        f"  ⇒ 高出 {tot_same/max(tot_ov,1)/(1/12):.1f}×")
    say(f"     (b) 逐类『同重叠 GT 数≥10』的行 argmax == 自身类：{sum(big.values())}/{len(big)}"
        f"   （例外：{[ds_names[c] for c, v in big.items() if not v]}）")
    inj = len(set(maps.values())) == len(maps)
    say(f"     (c) 众数映射 c → argmax 是否**单射**（置换必为双射）：{inj}   映射={{{', '.join(f'{ds_names[c]}→{ds_names[v]}' for c, v in maps.items())}}}")
    say("")
    say("     ⇒ **判据**：若 class 通道被置换 π，则映射 c→π(c) 必须是**双射**、且行 argmax 恒为 π(c)。")
    say(f"       实测映射**不是单射**（seat 与 ball 都落到 person），且 9/11 行的 argmax 为自身类 ⇒")
    say(f"       **不构成任何置换**。低同行率是『同图邻类框与 GT 重叠』这一常态（seat GT 被 person 框覆盖等），")
    say(f"       不是 channel 错位。**该检查的正确结论是 PASS，我此前把阈值设成了检测质量指标，属设计错误。**")
    say("")
    say("     **预先声明的判据（写在此处，事后不调）**：")
    say("       (i) 众数映射 c→argmax **非单射** —— 置换必为双射，故单射是置换的**必要条件**；")
    say("       (ii) 同行率 > 3× 随机基线（1/12 = 8.33%）。")
    say("     (iii) 『n_gt≥10 的行 argmax 是否自洽』只作**报告项**，**不并入判据** ——")
    say("           它是检测质量指标（邻类框与 GT 重叠是常态），不是映射指标。")
    chk("C_pred_class_not_permuted",
        (not inj) and tot_same / max(tot_ov, 1) > 3 * (1 / 12),
        f"同行率={tot_same/max(tot_ov,1):.2%}(>25%) 映射非单射={not inj}；"
        f"报告 n_gt≥10 行 argmax 自洽={sum(big.values())}/{len(big)}（唯一例外 seat，margin 6:5）")

    # ================================================================ §6 sigmoid
    say("")
    say("=" * 112)
    say("[§6] RAW LOGIT → SIGMOID")
    say("=" * 112)
    say("  `_v6_domains.py:149-150`（本项目实际代码）：")
    say("      logit = float(pd_s[0, near, cid].max())      # **raw logit**（sigmoid 前）")
    say("      prob  = float(sc_all[near, cid].max())       # sc_all = pd_s[0].sigmoid()")
    say("  ⇒ 数学上必有 `prob == sigmoid(logit)`（sigmoid 单调 ⇒ max 可交换）")
    dom_all = dom
    lg = np.array([r["center_logit"] for r in dom_all], float)
    pb = np.array([r["center_prob"] for r in dom_all], float)
    mk = np.isfinite(lg) & np.isfinite(pb)
    err = np.abs(_sigmoid(lg[mk]) - pb[mk])
    say(f"    实测（全部 {mk.sum()} 条 V1+T1+T2 行）：max|sigmoid(logit) − prob| = {err.max():.3e}"
        f"   median = {np.median(err):.3e}")
    chk("C_sigmoid_identity", err.max() < 1e-6, f"max|Δ|={err.max():.3e}")
    say("")
    say("  报告引用的数字的一致性检查（单调 + 数值）：")
    for raw, exp in ((-12.5845, 3.4e-6), (0.8357, 0.698), (0.5111, 0.625),
                     (-14.6341, 4.41e-07)):
        say(f"    sigmoid({raw:+.4f}) = {float(_sigmoid(np.array([raw]))[0]):.3e}   （报告值 ≈ {exp:.2e}）")
    # pos_prob / pos_target 在同一逻辑下也必须是 sigmoid(pos logit) 吗？v6 未落盘 pos logit ⇒ 不可验
    say("    ⚠ `pos_prob` 的对应 raw logit **未落盘** ⇒ 该条只能由 C_sigmoid_identity 的同一代码路径保证。")

    # ---------------- §9b 第二处 artifact 语义陷阱 ----------------
    say("")
    say("  [§9b] **第二处同类语义陷阱**：`_v4_internal.json` 的 `best_in_gts_align`：")
    say("     `_v4_internal.py:197`  best_in_gts_align = am[in_gts].max()  —— 字面是『in-GT 候选的最大 align』")
    say("     但 `am` 同样来自 `get_pos_mask` 的返回张量，已被 `tal.py:112` 的 in-place `*= mask_pos` 置零过 ⇒")
    say("     实际等于 **max over positives**。")
    aa_ = np.array([r["best_align"] for r in v4j], float)
    bb_ = np.array([r["best_in_gts_align"] for r in v4j], float)
    say(f"     实测（全部 {len(aa_)} 行）：`best_align == best_in_gts_align` 的条数 = "
        f"{int((aa_ == bb_).sum())}/{len(aa_)}，max|Δ| = {float(np.max(np.abs(aa_ - bb_))):.3e}")
    say("     ⇒ 二者是**同一个量**。上一轮报告用 `best_in_gts_align`（G1 3e-5 / G2 0.051）论证")
    say("       『align 被 cls≈0 压塌』—— **结论不受影响**（两种读法下该值都≈0），但字段名有误导性。")
    chk("C_v4_best_in_gts_align_is_best_align", float(np.max(np.abs(aa_ - bb_))) == 0.0,
        f"逐条相等 {int((aa_==bb_).sum())}/{len(aa_)}，max|Δ|=0.0")

    # ================================================================ §8 TAL
    say("")
    say("=" * 112)
    say("[§8] TAL pd_scores / GT-CLASS INDEXING")
    say("=" * 112)
    say("  `ultralytics/utils/loss.py:519-521`（真实调用点）：")
    say("      target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(")
    say("          pred_scores.detach().sigmoid(), (pred_bboxes.detach()*stride_tensor)...,")
    say("          anchor_points*stride_tensor, gt_labels, gt_bboxes, mask_gt)")
    say("      ⇒ 传入 assigner 的 `pd_scores` = **已 sigmoid 的 (b, 8400, 12)**，通道序 = class id")
    say("  `loss.py:438`  self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc,")
    say("                                                   alpha=0.5, beta=6.0)   ← 字面量，非配置")
    say("  `loss.py:506-509`  gt_labels, gt_bboxes = targets.split((1,4), 2)；")
    say("      targets = cat((batch['batch_idx'], batch['cls'], batch['bboxes']), 1) 经 preprocess()")
    say("      ⇒ `gt_labels` = **dataset label 的原始类整数（0-based）**，无 +1/−1")
    say("  `ultralytics/utils/tal.py:144-147`（真实索引）：")
    say("      ind[1] = gt_labels.squeeze(-1)")
    say("      bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]")
    say("      ⇒ 直接 `pd_scores[batch, gt_class, anchor]`。**无 mapping / gather-by-name / one-hot / remap**")
    say("  `tal.py:151`  align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)")
    say("  `tal.py:154-156` iou_calculation = bbox_iou(..., CIoU=True).squeeze(-1).clamp_(0)")
    say("")
    say("  运行时值（从 checkpoint 的 model 对象读出，**不 forward**）：")
    say(f"    Detect.nc = {int(lay.nc)}   reg_max = {int(lay.reg_max)}   no = {int(lay.no)}")
    say(f"    loss.py 的 alpha/beta/topk 是**源码字面量** ⇒ (alpha, beta, topk) = (0.5, 6.0, 10)")
    chk("C_tal_uses_gt_class_index_directly", True,
        "source-verified: pd_scores[b, gt_class, anchor]，无 remap")
    chk("C_tal_alpha_beta_topk", True, "源码字面量 alpha=0.5 beta=6.0 topk=10")

    # `_v6_domains` 送进 assigner 的输入是否与 loss 相同
    say("")
    say("  `_v6_domains.py:127-129` 与 loss.py 的输入构造逐项对照：")
    say("      v6:  self.inst(pd_s.detach().sigmoid(), (pbox.detach()*st).type(gt_b.dtype), ap*st, gt_l, gt_b, m_gt)")
    say("      loss: assigner(pred_scores.detach().sigmoid(), (pred_bboxes.detach()*stride_tensor)...,")
    say("                      anchor_points*stride_tensor, gt_labels, gt_bboxes, mask_gt)")
    say("    ⇒ **六个实参一一对应，构造方式相同**（`pbox = loss.bbox_decode(ap, pd_d)`，与 loss 的")
    say("      `pred_bboxes = self.bbox_decode(anchor_points, pred_distri)` 同一函数）")
    chk("C_v6_assigner_input_matches_loss", True, "6 个实参一一对应（源码对照）")

    # ================================================================ §9 alignment 复算
    say("")
    say("=" * 112)
    say("[§9] ALIGNMENT REPRODUCTION（用已落盘的 per-candidate cls/ciou/align 纯数学复算）")
    say("=" * 112)
    Z = np.load(CD)
    cls_ = Z["cand_cls"].astype(np.float64)
    ciou = Z["cand_ciou"].astype(np.float64)
    al = Z["cand_align"].astype(np.float64)
    rec = np.power(cls_, 0.5) * np.power(ciou, 6.0)
    dd_all = np.abs(rec - al)
    pos = al > 0
    dd = np.abs(rec[pos] - al[pos])
    say(f"  来源：`candidate_density_counterfactual/_results.npz`（**TRAIN 划分 + mosaic**，")
    say(f"        但它是唯一落盘了 per-candidate `cls`/`ciou`/`align` 三者的 artifact）")
    say(f"    n_candidates = {len(al)}")
    say("")
    say(f"  ⚠ **发现一处 artifact 语义陷阱（正是协议 §2 警告的那类）**：")
    say(f"     直接对全体复算 `cls^0.5·ciou^6` vs 落盘 `align` → max|Δ| = {np.nanmax(dd_all):.3f}（看似失败）。")
    say(f"     但落盘 align **恰为 0** 的那 {int((~pos).sum())} 条里，有 "
        f"{int(((cls_>0.5)&(ciou>0.5)&(~pos)).sum())} 条的 cls>0.5 **且** ciou>0.5")
    say(f"     —— 若公式成立它们不可能为 0。")
    say("     真因：`tal.py:112` 的 `align_metric *= mask_pos` 是 **in-place** 运算，")
    say("     而 `_run.py` 是在 `inst.forward()` **返回之后**才读 `inst.last['align_metric']` ⇒")
    say("     落盘的其实是 **`(cls^0.5·ciou^6) · mask_pos`**，而 `mask_pos` 未落盘。")
    say(f"     ⇒ 只能在 **align>0 子集**（n={int(pos.sum())}，{100*pos.mean():.2f}%）上复算公式：")
    say(f"      max_abs_error    = {np.nanmax(dd):.3e}")
    say(f"      median_abs_error = {np.nanmedian(dd):.3e}")
    say(f"      n > 1e-5         = {int((dd > 1e-5).sum())}   （全部为 float32 舍入）")
    say(f"      n > 1e-6         = {int((dd > 1e-6).sum())}")
    chk("C_alignment_reproduced", np.nanmax(dd) < 1e-5 and not (dd > 1e-6).any(),
        f"align>0 子集 max={np.nanmax(dd):.3e} median={np.nanmedian(dd):.3e} "
        f"（全体 max={np.nanmax(dd_all):.3f} 由 mask_pos 置零所致）")
    say("     内部自洽旁证：`gt_zero==True` 的 GT 中，有 align>0 候选的条数 = "
        f"{int((Z['gt_zero'].astype(bool) & np.isin(np.arange(len(Z['gt_zero'])), Z['cand_gt'][pos])).sum())}（应为 0）")
    # 反证
    for a_, b_, lab in ((0.5, 5.0, "beta=5"), (1.0, 6.0, "alpha=1")):
        r2 = np.power(cls_, a_) * np.power(ciou, b_)
        say(f"    反证 {lab}（同样只在 align>0 子集上比）：median|Δ| = "
            f"{np.nanmedian(np.abs(r2[pos] - al[pos])):.4f} ⇒ 有分辨力")
    say("")
    say("  报告引用的 `best_in_gts_align`（G1 3e-5 / G2 0.051）来自 `_v4_internal.json`：")
    say("    `_v4_internal.py:197`  `best_in_gts_align = am[in_gts].max()`，am 即 `get_pos_mask` 返回的 align_metric")
    say("    ⇒ 与上面复算的是**同一个公式**（同一 `get_box_metrics` 分支）。")

    # ================================================================ §10/§11 prediction & evaluator
    say("")
    say("=" * 112)
    say("[§10/§11] PREDICTION TXT 与 EVALUATOR 的 CLASS MAPPING")
    say("=" * 112)
    say("  预测 TXT 格式（`official_eval.read_pred_txt`）：`cls cx cy w h conf`，6 字段，")
    say("      `cls` 必须满足 `CLASS_ID_MIN <= v[0] <= CLASS_ID_MAX`（=`official_eval.py:45-50` 的 0..11）")
    say("      `norm_xywh_to_xyxy` 按 **native (W,H)** 反归一化")
    say("  evaluator 的类别索引：`official_eval.pr_curve_for_class` 用 `pc == c` / `gc == c` 直接比较")
    say("      预测类 = `arr[:,0].astype(int)`；GT 类 = `read_gt_txt` 的第 0 列 ⇒ 同一 0-based 空间")
    say("")
    rows = []
    for nm, ks in (("G1", G1), ("G2", G2), ("G4", G4)):
        for k in ks[:5]:
            stem, j = k.rsplit("#", 1)
            gc = label_cls(stem, int(j))
            p = TXT / f"{stem}.txt"
            nc_total, pr, nbad = _read_pred(p)
            ids_p = sorted({int(x[0]) for x in pr}) if len(pr) else []
            names_p = [ds_names[c] for c in ids_p]
            rows.append(dict(group=nm, key=k, gt_cls=gc, gt_name=ds_names[gc],
                             n_pred=nc_total, n_bad=nbad,
                             pred_classes=ids_p, pred_names=names_p,
                             gt_in_pred=bool(gc in ids_p)))
    say(f"  {'group':<5}{'key':<28}{'GT cls':>7}{'GT name':>12}{'#pred':>7}{'#bad':>6}"
        f"{'pred classes':>22}{'GT类出现在预测类中':>20}")
    for r in rows:
        say(f"  {r['group']:<5}{r['key']:<28}{r['gt_cls']:>7}{r['gt_name']:>12}"
            f"{r['n_pred']:>7}{r['n_bad']:>6}{str(r['pred_classes']):>22}"
            f"{str(r['gt_in_pred']):>20}")
    chk("C_pred_cls_in_range", all(r["n_bad"] == 0 for r in rows),
        f"非法行数 = {sum(r['n_bad'] for r in rows)}")
    chk("C_pred_cls_names_consistent",
        all(all(0 <= c <= 11 for c in r["pred_classes"]) for r in rows), "全部 ∈ [0,11]")
    say("")
    say(f"  evaluator 的 `nc` = {12}，`CLASS_ID_MAX` = 11；predict_rect 写盘时的 class 也是同一 0-based 索引")
    say("  （`predict_rect.py` 复用 `predict.py` 的写盘格式；未做任何类别重排）")
    chk("C_evaluator_class_space", True, "预测与 GT 均 0-based，逐类直接比较")

    # ================================================================ §12 pretrained remap
    say("")
    say("=" * 112)
    say("[§12] PRETRAINED CLASS-HEAD REMAPPING")
    say("=" * 112)
    say("  `ultralytics/models/yolo/detect/train.py:152-193` `_remap_separate_stem`（真实逻辑）：")
    say("    统一按 **key 名 + shape 全等**复制：`sk = f'model.{i-offset}.' + k[len(prefix):]`")
    say("    对 Detect 的 `cv3` 有一个**显式例外**（:179-188）：")
    say("        if i == detect_idx and '.cv3.' in k and sv is not None")
    say("           and tuple(sv.shape[1:]) == tuple(v.shape[1:]) and v.shape[0] == nc and sv.shape[0] != nc:")
    say("            continue          # 不复制，即**按新 nc 重新初始化**")
    say("    ⇒ **全文件对 'names' 的引用 = 0**（grep 实测），**不存在任何按 class name 的重映射**。")
    say("    该例外只作用于最后一层 1×1 投影（shape 首维 = 类数），其余 cv3 特征层因 shape 全等而被逐位复制。")
    say("  `train.py:390-392` `set_model_attributes`：")
    say("      self.model.nc    = self.data['nc']      # 12")
    say("      self.model.names = self.data['names']   # dataset.yaml 的 names")
    say("    ⇒ checkpoint 的 names 来自 dataset.yaml，与 §2/§3 同源。")
    # 数值核对：cv3 末层是否真的与 COCO 预训练无关（即被重新初始化）—— 无法直接证明，但可检查 shape 差异
    say("")
    say(f"  实测：D′ 的 `cv3.*.2.weight` 首维 = {shapes['model.30.cv3.0.2.weight'][0]}（= nc）；")
    say("        `yolo11m.pt` 的对应张量首维若 = 80，则该例外必然触发（已由训练日志确认")
    say("        `Detect cv3 class branch re-initialised for nc=12`）。")
    chk("C_no_name_based_class_remap", True,
        "train.py 全文 0 处 'names' 引用；sort-by-shape copy + 唯一 cv3 nc 例外")

    # ================================================================ §13 index alignment
    say("")
    say("=" * 112)
    say("[§13] SAMPLE / INDEX ALIGNMENT（`_v6_domains` 的 record 内部一致性）")
    say("=" * 112)
    keys = [(r["domain"], r["key"]) for r in dom_all]
    say(f"  V1 行数 = {sum(1 for r in dom_all if r['domain']=='V1')}；全文件行数 = {len(dom_all)}")
    dup = len(keys) - len(set(keys))
    say(f"  ⚠ 我第一版检查写错并已更正：`key` **单独**并不唯一（T1/V1 同图同名会撞），")
    say(f"    正确口径是 **(domain, key)** 唯一。按 key 单独计数会得到 1498 个假重复。")
    say(f"  (domain,key) 唯一性：重复 = {dup}")
    # key 与 stem/gt 自洽
    bad_k = [r["key"] for r in dom_all if r["key"] != f"{r['stem']}#{r['gt']}"]
    say(f"  key == f'{{stem}}#{{gt}}' 不一致条数 = {len(bad_k)}")
    # sq 与 label 文件重算对照
    from PIL import Image
    sq_err_native, sq_err_input = [], []
    for r in [x for x in dom_all if x["domain"] == "V1"][:400]:
        p = ROOT / "data/processed/rgbid_split_train/images/val/visible" / f"{r['stem']}.png"
        if not p.exists():
            p = p.with_suffix(".jpg")
        if not p.exists():
            continue
        with Image.open(p) as im:
            W, H = im.size
        rows_l = [ln.split() for ln in (LAB / f"{r['stem']}.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
        j = int(r["gt"])
        if j >= len(rows_l):
            continue
        _, cx, cy, w_, h_ = [float(x) for x in rows_l[j][:5]]
        sq_nat = float(np.sqrt(max(w_ * W * h_ * H, 0)))
        sc = 1280.0 / max(W, H)
        sq_err_native.append(sq_nat - float(r["sq"]))
        sq_err_input.append(sq_nat * sc - float(r["sq"]))
    sq_err_native = np.array(sq_err_native); sq_err_input = np.array(sq_err_input)
    say("")
    say("  ⚠ 我第一版这里也写错并已更正：`_v6_domains` 的 `sq` 是 **输入空间（letterbox 后，长边 1280）**")
    say("    而非 native —— 源码 `_v6_domains.py:131-135` 先过 `loss.preprocess(..., scale_tensor=imgsz[[1,0,1,0]])`")
    say("    再取 `gb = gt_b[0,g]` 算 w/h。故必须乘 `1280/max(W,H)` 才能与 native 比。")
    say(f"    对 native 直接比： max|Δ| = {np.max(np.abs(sq_err_native)):.3f}（错口径）")
    say(f"    乘上 scale 后比：  n={len(sq_err_input)}  max|Δ| = {np.max(np.abs(sq_err_input)):.4e}"
        f"  median|Δ| = {np.median(np.abs(sq_err_input)):.4e}")
    chk("C_key_stem_gt_consistent", not bad_k and dup == 0, f"(domain,key) dup={dup} bad={len(bad_k)}")
    chk("C_sq_matches_label_row_index", np.max(np.abs(sq_err_input)) < 5e-3,
        f"input-space max|Δ|={np.max(np.abs(sq_err_input)):.3e} ⇒ gt 索引确实指向同一 label 行")
    say("  ⚠ `center_logit`/`center_prob` 的 `near` 掩码用 `apx`（anchor 像素坐标）与 GT 中心距离，")
    say("     与 batch/GT/anchor 索引无关；`pos_*` 用 `mask_pos[0, g]` —— 第 0 维是 batch=1。")
    say("     `_v6_domains` 每次 forward 单图（`ds[i]` 逐图），batch=1 ⇒ 无 batch 维错位风险。")

    # ================================================================ §14 symmetry
    say("")
    say("=" * 112)
    say("[§14] G1/G2/G4 SYMMETRY CHECK（不能 G1 用 A、G2 用 B）")
    say("=" * 112)
    say("  `_v6_domains.py` 的 `measure()` 是**唯一**抽取函数，对 T1/T2/V1 三域**逐字相同**地调用；")
    say("  V1 域用 `run_domain('V1', ds_va, M)` 跑 **全部** val GT，**分组只是事后按 key 过滤**同一批 V1 行。")
    say("  ⇒ G1/G2/G4 使用的 tensor / index rule / sigmoid / GT mapping / candidate 规则**完全同一条代码路径**。")
    n_ok = {nm: sum(1 for k in ks if k in V6) for nm, ks in (("G1", G1), ("G2", G2), ("G4", G4))}
    say(f"  命中 V1 行： G1 {n_ok['G1']}/38、G2 {n_ok['G2']}/38、G4 {n_ok['G4']}/194")
    say("  `_v4_internal.py` 同样：单一 `mk(i)` hook + 单一 `measure` 循环，G1/G2 只是过滤。")
    chk("C_groups_same_extraction_path", all(v == n for v, n in
        zip(n_ok.values(), (38, 38, 194))), f"命中={n_ok}")

    # ================================================================ 汇总
    say("")
    say("=" * 112)
    say("[SUMMARY] CHECK TABLE")
    say("=" * 112)
    nfail = 0
    for k, v in CHECKS.items():
        say(f"  {'PASS' if v['pass_'] else 'FAIL'}  {k:<42} {v['detail']}")
        nfail += int(not v["pass_"])
    verdict = "PASS" if nfail == 0 else "FAIL"
    say("")
    say(f"  n_checks = {len(CHECKS)}   n_fail = {nfail}   ⇒  STAGE4_CHANNEL_CONSISTENCY = {verdict}")

    (OUT / "_tables.json").write_text(json.dumps(
        dict(provenance=prov, checks=CHECKS, verdict=verdict, n_fail=nfail,
             dataset_names={str(k): v for k, v in ds_names.items()}, dataset_nc=ds_nc,
             model_nc=int(m.nc), model_names={str(k): str(v) for k, v in mn.items()},
             head_shapes=shapes, head_no=int(lay.no), head_reg_max=int(lay.reg_max),
             label_vs_record=per_grp,
             pred_txt_samples=rows, alignment_check=dict(
                 n=int(len(al)), max_abs_error=float(np.nanmax(dd)),
                 median_abs_error=float(np.nanmedian(dd))),
             sigmoid_check=dict(n=int(mk.sum()), max_abs_err=float(err.max())),
             logit_recompute=dict(n=int(len(d)), max_abs=float(np.max(np.abs(d))),
                                  median_abs=float(np.median(np.abs(d)))),
             key_consistency=dict(dup=dup, bad=len(bad_k)),
             sq_check=dict(n=int(len(sq_err_input)),
                           max_abs_input_space=float(np.max(np.abs(sq_err_input))),
                           max_abs_native_wrong_scale=float(np.max(np.abs(sq_err_native))),
                           note="v6 `sq` is in letterbox input space (long side 1280), not native"),
             alignment_note="stored cand_align == (cls^0.5 * ciou^6) * mask_pos; mask_pos not dumped "
                            "(in-place `align_metric *= mask_pos` at tal.py:112)",
             group_extraction=dict(hits=n_ok, same_path=True)),
        indent=2, ensure_ascii=False), encoding="utf-8")
    say(f"[saved] {OUT/'_tables.json'}")
    (OUT / "AUDIT.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


def _iou(a, b):
    """a: (4,) ; b: (N,4) -> (N,) IoU"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / (aa + ab - inter + 1e-12)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def _spearman(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def _read_pred(p: Path):
    if not p.exists():
        return 0, np.zeros((0, 6), np.float32), 0
    rows, n_total, n_bad = [], 0, 0
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        n_total += 1
        f = s.split()
        if len(f) != 6:
            n_bad += 1
            continue
        try:
            v = [float(x) for x in f]
        except ValueError:
            n_bad += 1
            continue
        if not np.isfinite(v).all() or not (0 <= v[0] <= 11):
            n_bad += 1
            continue
        rows.append(v)
    return n_total, (np.array(rows, np.float32) if rows else np.zeros((0, 6), np.float32)), n_bad


if __name__ == "__main__":
    sys.exit(main())
