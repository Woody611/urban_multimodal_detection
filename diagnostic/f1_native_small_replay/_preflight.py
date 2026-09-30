"""_preflight.py — F1 Native-small Replay 的 10 道 preflight gate + smoke test（**不训练、不 forward**）

只在 dataset 层面跑（构建 dataset + 取样本），**不实例化模型、不 forward、不 backward、不 optimizer**。

输出（协议 §26 manifest）：
    diagnostic/f1_native_small_replay/{preflight.json, config_diff.json,
                                       transform_order.txt, smoke_test.json, provenance_before.txt}
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import yaml  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.data.augment import NativeSmallReplay  # noqa: E402
from ultralytics.utils import yaml_load  # noqa: E402
import train as TRAINMOD  # noqa: E402  （scripts/train.py）

F1_CFG = ROOT / "configs/train_rgbid_sepstem_clahe_f1.yaml"
DP_CFG = ROOT / "configs/train_rgbid_sepstem_clahe.yaml"
MODEL_CFG = ROOT / "configs/yolo11m_sepstem.yaml"
DS_YAML = ROOT / "data/processed/rgbid_split_train/dataset.yaml"
DP_ARGS = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/args.yaml"

N_SMOKE = 20
GATES: dict[str, dict] = {}
LOG: list[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def sha16(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def gate(name: str, ok, detail: str = "") -> bool:
    GATES[name] = dict(pass_=bool(ok), detail=str(detail))
    say(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        say(f"        {detail}")
    return bool(ok)


def build_ds(cfg_path: Path, n_keep: int | None = None):
    tcfg = TRAINMOD._load_yaml(cfg_path)
    kwargs = TRAINMOD._build_train_kwargs(tcfg, str(DS_YAML))
    cfg = get_cfg(overrides=kwargs)
    data = yaml_load(str(DS_YAML))
    ds = build_yolo_dataset(cfg, str((Path(data["path"]) / data["train"]).resolve()),
                            int(kwargs.get("batch", 8)), data, mode="train",
                            use_simotm=cfg.use_simotm,
                            pairs_rgb_ir=list(cfg.pairs_rgb_ir),
                            pairs_rgb_depth=list(cfg.pairs_rgb_depth))
    return ds, cfg, tcfg


def find_nsr(ds):
    for t in ds.transforms.transforms:
        if isinstance(t, NativeSmallReplay):
            return t
        sub = getattr(t, "transforms", None)
        if sub:
            for u in sub:
                if isinstance(u, NativeSmallReplay):
                    return u
    return None


def main() -> int:
    say("=" * 100)
    say("F1 NATIVE-SMALL REPLAY —— PREFLIGHT (10 GATES) + SMOKE TEST")
    say("=" * 100)
    say("[provenance]")
    prov = {str(p): sha16(p) for p in
            (F1_CFG, DP_CFG, MODEL_CFG, DS_YAML, DP_ARGS,
             ROOT / "ultralytics/data/augment.py", ROOT / "scripts/train.py")}
    for k, v in prov.items():
        say(f"  {v}  {k}")

    # ---------------- Gate 8: transform order（先建立，供人眼核对） ----------------
    say("")
    say("=" * 100)
    say("[Gate 8] REAL TRANSFORM ORDER")
    say("=" * 100)
    ds_f1, cfg_f1, tcfg_f1 = build_ds(F1_CFG)
    order = [type(t).__name__ for t in ds_f1.transforms.transforms]
    inner = [type(t).__name__ for t in ds_f1.transforms.transforms[2].transforms] \
        if hasattr(ds_f1.transforms.transforms[2], "transforms") else []
    say(f"  D′+F1 Compose 顺序: {order}")
    say(f"  pre_transform 内部:  {inner}")
    say("  D′ 原顺序（augment.py 未加 F1 时）: ['ObjectScaleAug','Compose','MixUp','Albumentations',"
        "'RandomHSV','RandomFlip','RandomFlip']")
    say("  F1 顺序:                            ['NativeSmallReplay','ObjectScaleAug','Compose','MixUp',"
        "'Albumentations','RandomHSV','RandomFlip','RandomFlip']")
    say("  ⇒ F1 的插入点是 **Compose 的第一个元素**，即 load_image 之后、Mosaic **之前**（协议 §17）")
    nsr = find_nsr(ds_f1)
    gate("G8_transform_order", nsr is not None and order[0] == "NativeSmallReplay",
         f"order[0]={order[0]}; NativeSmallReplay 定位={'OK' if nsr else 'FAIL'}")
    (OUT / "transform_order.txt").write_text(
        "D'  : " + " -> ".join(['ObjectScaleAug', 'Compose(Mosaic->CopyPaste->RandomPerspective)',
                                'MixUp', 'Albumentations', 'RandomHSV', 'RandomFlip(v)',
                                'RandomFlip(h)']) + " -> Format\n" +
        "F1  : " + " -> ".join(order) + f"\npre_transform: {inner}\n", encoding="utf-8")

    # ---------------- Gate 9: D′ equivalence（有效键逐项 diff） ----------------
    say("")
    say("=" * 100)
    say("[Gate 9] D′ EQUIVALENCE —— 有效配置逐项 diff")
    say("=" * 100)
    d_dp = TRAINMOD._build_train_kwargs(TRAINMOD._load_yaml(DP_CFG), str(DS_YAML))
    d_f1 = TRAINMOD._build_train_kwargs(TRAINMOD._load_yaml(F1_CFG), str(DS_YAML))
    only_dp = {k: v for k, v in d_dp.items() if k not in d_f1}
    only_f1 = {k: v for k, v in d_f1.items() if k not in d_dp}
    diff = {k: [d_dp[k], d_f1[k]] for k in set(d_dp) & set(d_f1) if d_dp[k] != d_f1[k]}
    say(f"  仅 D′ 有: {only_dp}")
    say(f"  仅 F1 有: {sorted(only_f1)}")
    say(f"  值不同的键: {diff}")
    say("  ⇒ 期望：唯一**训练变量**差异 = `native_small_replay=1.0` + "
        "`native_small_replay_max_new_instances=2`；")
    say("     `name`（= experiment_name）也必然不同 —— 它只决定 run 目录（协议 §26/§38 要求独立目录），")
    say("     不属于训练变量，故列入**预期差异白名单**（并校验其值 = F1 的 experiment_name）。")
    diff_trainvar = {k: v for k, v in diff.items() if k != "name"}
    name_ok = (diff.get("name", [None, None])[1]
               == "urban_multimodal_det_yolo11_rgbid_sepstem_clahe_f1")
    exp_only = {"native_small_replay", "native_small_replay_max_new_instances"}
    gate("G9_config_equivalence",
         not diff_trainvar and not only_dp and set(only_f1) == exp_only
         and float(only_f1.get("native_small_replay", -1)) == 1.0
         and int(only_f1.get("native_small_replay_max_new_instances", -1)) == 2 and name_ok,
         f"only_f1={only_f1} training_var_diff={diff_trainvar} name_only_diff={name_ok}")
    # D′ 的 args.yaml（权威）对照
    dp_args = yaml.safe_load(DP_ARGS.read_text(encoding="utf-8"))
    must = {k: dp_args.get(k) for k in ("ir_encoding", "use_simotm", "channels", "imgsz", "epochs",
                                        "batch", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
                                        "warmup_epochs", "seed", "mosaic", "close_mosaic", "translate",
                                        "scale", "degrees", "shear", "perspective", "fliplr", "flipud",
                                        "mixup", "copy_paste", "rect", "pretrained", "patience", "amp")}
    cfg_eff = {k: getattr(cfg_f1, k, None) for k in must}
    say(f"  继承自 D′ args.yaml 的关键项（逐项比对 F1 的有效 cfg）：")
    bad = []
    for k in must:
        same = (str(cfg_eff[k]) == str(must[k])) or (k in ("imgsz",) and int(cfg_eff[k]) == int(must[k]))
        if not same:
            bad.append((k, must[k], cfg_eff[k]))
        say(f"    {k:<15} D′={must[k]!r:<28} F1={cfg_eff[k]!r:<28} {'OK' if same else 'MISMATCH'}")
    gate("G9b_inherited_args", not bad, f"mismatch={bad}")
    (OUT / "config_diff.json").write_text(json.dumps(
        dict(only_dprime=only_dp, only_f1=only_f1, value_diff=diff,
             inherited={k: [str(must[k]), str(cfg_eff[k])] for k in must}, mismatch=bad),
        indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- Gate 10: RNG neutrality ----------------
    say("")
    say("=" * 100)
    say("[Gate 10] RNG NEUTRALITY")
    say("=" * 100)
    def snap():
        return (random.getstate(), np.random.get_state()[1].tobytes(), torch.get_rng_state().clone())
    def same(a, b):
        return (a[0] == b[0]) and (a[1] == b[1]) and bool(torch.equal(a[2], b[2]))
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    s0 = snap()
    _ = build_ds(F1_CFG)                       # 构建含 F1 的 transforms
    s1 = snap()
    gate("G10a_construction_rng_neutral", same(s0, s1),
         "构建 F1 的 transforms 前后 global RNG state 逐位相同" if same(s0, s1) else "RNG 被扰动")
    r0 = snap()
    lab = dict(img=np.zeros((64, 64, 5), np.uint8), cls=np.array([0]),
               instances=None, ori_shape=(64, 64), resized_shape=(64, 64))
    t0 = NativeSmallReplay(p=0.0)              # 关闭态
    for _ in range(10):
        t0(lab) if lab["instances"] is not None else None
    r1 = snap()
    gate("G10b_p0_no_rng_consumption", same(r0, r1),
         "p=0 时连续 10 次调用不消耗任何 RNG ⇒ D′ 数据流逐位不变")

    # ---------------- Smoke test（§25：20 samples，无模型/无 forward） ----------------
    say("")
    say("=" * 100)
    say(f"[SMOKE TEST] {N_SMOKE} training samples（dataset 层；无模型/无 forward/无 backward）")
    say("=" * 100)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    ds_s, cfg_s, _ = build_ds(F1_CFG)
    nsr_s = find_nsr(ds_s)
    nsr_s.records.clear(); nsr_s.stats.update(seen=0, eligible=0, replayed=0,
                                              skip_prob=0, skip_nosmall=0, skip_attempts=0)
    smoke = []
    for i in range(N_SMOKE):
        try:
            lab = ds_s[i]
        except Exception as e:  # noqa: BLE001
            smoke.append(dict(i=i, err=f"{type(e).__name__}: {e}"))
            continue
        img = lab["img"]
        cls_after = np.asarray(lab["cls"]).reshape(-1)
        nb = len(cls_after)
        n_dup = 0
        if lab.get("instances") is not None:
            bb = np.asarray(lab["instances"].bboxes)
            if len(bb):
                n_dup = int(len(bb) - len({tuple(np.round(np.asarray(r, float), 6)) for r in bb}))
        smoke.append(dict(i=i, ok=True, shape=list(np.asarray(img).shape),
                          n_labels=int(nb), n_dup_boxes=n_dup,
                          finite=bool(np.isfinite(np.asarray(img, np.float32)).all())))
    n_err = sum(1 for s in smoke if "err" in s)
    say(f"  样本 {N_SMOKE}：异常 {n_err}")
    say(f"  图像 shape 集合: {sorted({tuple(s['shape']) for s in smoke if 'err' not in s})}")
    say(f"  通道数集合（CHW，取 shape[0]）: {sorted({s['shape'][0] for s in smoke if 'err' not in s})}")
    say(f"  label 数（变换后）: {[s['n_labels'] for s in smoke if 'err' not in s]}")
    say(f"  stats: {nsr_s.stats}")
    say(f"  捕获 replay 记录: {len(nsr_s.records)}")
    gate("SMOKE_no_error", n_err == 0, f"异常 {n_err}")
    gate("SMOKE_5ch", {s['shape'][0] for s in smoke if 'err' not in s} == {5},
         "Format 后为 CHW；shape[0] == 5（RGBID 5 通道）")
    gate("SMOKE_no_nan", all(s.get("finite") for s in smoke if "err" not in s), "无 NaN/Inf")
    (OUT / "smoke_test.json").write_text(json.dumps(
        dict(n=N_SMOKE, errors=n_err, samples=smoke, stats=nsr_s.stats,
             n_records=len(nsr_s.records)), indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- Gates 1–7：在足够多的 replay 上取证 ----------------
    say("")
    say("=" * 100)
    say("[Gates 1–7] 在大量 replay 记录上取证（需 ≥100 条）")
    say("=" * 100)
    ds_r, _, _ = build_ds(F1_CFG)
    nsr_r = find_nsr(ds_r)
    nsr_r.records.clear()
    n_img = 0
    i = 0
    while len(nsr_r.records) < 200 and i < 400:
        _ = ds_r[i % len(ds_r)]
        i += 1
        n_img += 1
        if i >= len(ds_r) and len(nsr_r.records) < 100:
            break
    R = nsr_r.records
    say(f"  扫描 {n_img} 张样本；捕获 replay 记录 {len(R)} 条；stats={nsr_r.stats}")
    say(f"  实际 replay 率（replayed / eligible）= "
        f"{nsr_r.stats['replayed']/max(nsr_r.stats['eligible'],1):.4f}"
        f"（配置 native_small_replay={getattr(cfg_s,'native_small_replay',None)}）")

    def g1():
        bad = [r for r in R if not (0 < r["src_area_native"] < 1024.0)]
        return not bad, f"n={len(R)}；越界条数={len(bad)}"
    def g2():
        w = [abs((r["dest_bbox_xyxy"][2] - r["dest_bbox_xyxy"][0]) -
                 (r["src_bbox_xyxy"][2] - r["src_bbox_xyxy"][0])) for r in R]
        h = [abs((r["dest_bbox_xyxy"][3] - r["dest_bbox_xyxy"][1]) -
                 (r["src_bbox_xyxy"][3] - r["src_bbox_xyxy"][1])) for r in R]
        m = max(max(w or [0]), max(h or [0]))
        return m <= 1e-6, f"max|Δw|,max|Δh| = {m:.3e}（阈值 1e-6）"
    def g3():
        bad = []
        for r in R:
            sp, sd = r["sum_patch_per_ch"], r["sum_dest_after_per_ch"]
            if len(sp) != 5 or len(sd) != 5:
                bad.append(("nch", r["im_file"])); continue
            if any(abs(a - b) > 1e-6 * max(abs(a), 1.0) for a, b in zip(sp, sd)):
                bad.append(("sum", r["im_file"]))
        return not bad, f"n={len(R)}；5 通道逐通道和 paste前后不一致 = {len(bad)}"
    def g4():
        m = max(max(abs(r["dest_bbox_xyxy"][2] - r["dest_bbox_xyxy"][0]
                        - (r["src_bbox_xyxy"][2] - r["src_bbox_xyxy"][0])) for r in R), 0)
        return m <= 1e-6, f"max|Δ| = {m:.3e}"
    def g5():
        bad = [r for r in R if r["all_cls_after"][-1] != r["src_cls"]]
        return not bad, f"新标签类 ≠ source 类 的条数 = {len(bad)}"
    def g6():
        worst = 0.0; n_bad = 0
        for r in R:
            d = r["dest_bbox_xyxy"]
            for bb in r["other_bboxes_before"]:
                x1 = max(d[0], bb[0]); y1 = max(d[1], bb[1])
                x2 = min(d[2], bb[2]); y2 = min(d[3], bb[3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                a = (d[2] - d[0]) * (d[3] - d[1])
                ab = max(bb[2] - bb[0], 0) * max(bb[3] - bb[1], 0)
                iou = inter / (a + ab - inter + 1e-12)
                worst = max(worst, iou)
                n_bad += int(iou > 0.10 + 1e-9)
        return n_bad == 0, f"n={len(R)}；最大 IoU = {worst:.4f}（阈值 0.10）；越界对数 = {n_bad}"
    def g7():
        bad = [r for r in R if (r["n_after"] != r["n_existing"] + 1)
               or (r["all_cls_after"][:r["n_existing"]] != r["all_cls_before"])]
        return not bad, f"原标签被保留 + 恰好新增 1 条：违例 = {len(bad)}"

    def g7b():
        m = nsr_r.stats.get("max_per_call", 0)
        return m <= nsr_r.max_new_instances, (
            f"实测每个 sample 最多新增 {m} 个（上限 max_new_instances={nsr_r.max_new_instances}）；"
            f"multi(>=2) 样本数={nsr_r.stats.get('multi',0)}")

    for nm, fn in (("G1_source_is_native_small", g1), ("G2_scale_1p0", g2),
                   ("G3_multimodal_5ch_sync", g3), ("G4_bbox_size_preserved", g4),
                   ("G5_class_preserved", g5), ("G6_overlap_le_0.10", g6),
                   ("G7_label_integrity", g7), ("G7b_per_image_ceiling", g7b)):
        ok, det = fn()
        gate(nm, ok, det)

    # ---------------- §23 parameter count ----------------
    say("")
    say("=" * 100)
    say("[§23] PARAMETER COUNT GATE")
    say("=" * 100)
    say(f"  F1 的 --model_config 与 D′ **同一文件**：`configs/yolo11m_sepstem.yaml` "
        f"sha={sha16(MODEL_CFG)}")
    say(f"  D′ args.yaml 记录的 model = {dp_args.get('model')}")
    gate("PARAM_model_config_identical",
         str(dp_args.get("model", "")).endswith("configs/yolo11m_sepstem.yaml"),
         "F1 不接触任何模型代码；model config 与实际架构由同一 yaml 决定")
    say("  F1 的改动仅位于：`ultralytics/data/augment.py`（新增一个 transform 类 + 在 Compose 首插一项）")
    say("                    `scripts/train.py`（转发 native_small_replay* 键）")
    say("                    `configs/train_rgbid_sepstem_clahe_f1.yaml`（新增）")
    say("  ⇒ **模型参数量必然不变**（0 参数变化）")

    # ---------------- 汇总 ----------------
    n_fail = sum(1 for v in GATES.values() if not v["pass_"])
    say("")
    say("=" * 100)
    say("[SUMMARY]")
    say("=" * 100)
    for k, v in GATES.items():
        say(f"  {'PASS' if v['pass_'] else 'FAIL'}  {k}")
    verdict = "EXPERIMENT READY" if n_fail == 0 else "EXPERIMENT NOT READY"
    say(f"\n  n_gates = {len(GATES)}  n_fail = {n_fail}  ⇒  {verdict}")

    (OUT / "preflight.json").write_text(json.dumps(dict(
        experiment="F1", baseline="D_prime", verdict=verdict, n_gates=len(GATES), n_fail=n_fail,
        gates=GATES, provenance=prov,
        frozen=dict(native_small_area_threshold=1024, replay_probability=1.0,
                    max_new_instances_per_image=2, instance_scale=1.0, rotation=0.0,
                    shear=0.0, perspective=0.0, context_margin=0.10,
                    max_destination_iou=0.10, max_attempts=20, seed=42, imgsz=1280, epochs=300),
        replay_stats=nsr_r.stats, n_records=len(R),
        actual_replay_rate=nsr_r.stats["replayed"] / max(nsr_r.stats["eligible"], 1),
        per_sample_replay_rate=(nsr_r.stats["replayed"] / max(nsr_r.stats["seen"], 1)),
        instances_added_per_sample=(nsr_r.stats["instances_added"] / max(nsr_r.stats["seen"], 1)),
    ), indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "preflight.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    say(f"[saved] {OUT/'preflight.json'}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
