"""_run_probe300_ema.py — Full-300 **EMA** trajectory probe 的云端训练入口（本机无 GPU）。

与旧 `_run_probe300.py` 的关键差异（三处，旧版已判 INVALID）：
  · probe target = `trainer.ema.ema`（旧版 `trainer.model`）
  · cell 坐标 = `medium_cells_canvas.json`（旧版漏 letterbox pad 的 `medium_cells.json`）
  · mode = 纯 `model.eval()`（旧版 `eval backbone + head.train()`）

启动顺序（任一 FAIL 即中止，§13）：
  1) D′ provenance 硬断言
  2) cells 文件存在 + legacy regression 已通过
  3) **in-process P4 gate**：在真实 ema.ema 上跑 self_check，非侵入 + 恒等式
  4) 才开始 300 epoch

用法（云端，仓库根目录）：
  python -X utf8 scripts/preflight_check_train_config.py \
      --model_config configs/yolo11m_sepstem.yaml \
      --train_config configs/train_rgbid_sepstem_clahe.yaml --expect-ir-encoding clahe
  python -X utf8 diagnostic/full300_trajectory_probe/_run_probe300_ema.py
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from ultralytics import YOLO  # noqa: E402
from _probe300_ema import CELLS, OBSERVE_0B, EmaProbe  # noqa: E402

RUN_NAME = "urban_multimodal_det_yolo11_rgbid_sepstem_clahe_FULL300_EMA"

spec = importlib.util.spec_from_file_location("tr", ROOT / "scripts/train.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

train_cfg = tr._load_yaml(ROOT / "configs/train_rgbid_sepstem_clahe.yaml")
dataset_cfg = tr._load_yaml(ROOT / "configs/dataset.yaml")
data_path = tr._split_train_val_rgbid(dataset_cfg, float(train_cfg.get("val_ratio", 0.2)),
                                      int(train_cfg.get("seed", 42)))
kwargs = tr._build_train_kwargs(train_cfg, data_path)
_DEF = tr._load_yaml(ROOT / "ultralytics/cfg/default.yaml")
eff = lambda k: kwargs[k] if k in kwargs else _DEF.get(k)
kwargs.update(resume=False, exist_ok=False, name=RUN_NAME)

# ---------- §1 provenance 硬断言（与 D′ 逐项相同） ----------
assert kwargs["epochs"] == 300 and kwargs["patience"] == 0
assert kwargs["seed"] == 42 and kwargs["batch"] == 8 and kwargs["imgsz"] == 1280
assert kwargs["optimizer"] == "SGD" and kwargs["lr0"] == 0.005 and kwargs["cos_lr"] is True
assert kwargs["resume"] is False and kwargs.get("cls_pw") is None
assert eff("rect") is False and eff("mosaic") == 1.0 and eff("close_mosaic") == 10
assert eff("box") == 7.5 and eff("cls") == 0.5 and eff("dfl") == 1.5
assert eff("ir_encoding") == "clahe", f"ir_encoding={eff('ir_encoding')}"
assert eff("use_simotm") == "RGBID" and eff("channels") == 5
assert not (ROOT / str(kwargs.get("project", "runs")) / RUN_NAME).exists(), "目标目录已存在"

# ---------- 陈旧 npz 污染门 ----------
# `_analyze300.py` 用 OUT.glob("epoch_*.npz") 读**全部** npz。若目录里还留着上一轮
# （raw-model / 错误坐标，已判 INVALID）的 epoch_NNN.npz，后处理会把新旧混在一起算出
# 看似合理的结论。本目录实测残留 18 个旧 npz（0..289）。启动前必须清空。
_stale = sorted(HERE.glob("epoch_*.npz"))
assert not _stale, (
    f"目录内残留 {len(_stale)} 个旧 epoch_*.npz（{_stale[0].name} .. {_stale[-1].name}），"
    f"它们来自已作废的 raw probe，会污染 _analyze300.py。\n"
    f"请先归档：mkdir -p {HERE}/_old_raw_probe && mv {HERE}/epoch_*.npz "
    f"{HERE}/trajectory_*.csv {HERE}/mechanism_report.md {HERE}/trajectory_decomposition.json "
    f"{HERE}/_old_raw_probe/")
for _f in ("trajectory_summary.csv", "trajectory_per_gt.csv", "trajectory_tal.csv",
           "trajectory_classifier_drift.csv", "trajectory_representation_drift.csv",
           "trajectory_decomposition.csv", "trajectory_metadata.json"):
    assert not (HERE / _f).exists(), f"目录内残留旧产物 {_f} —— 同上，请先归档"

# ---------- §2 cells / legacy regression 门 ----------
assert CELLS.exists(), f"缺 cells 文件：{CELLS}（先跑 diagnostic/p3_trajectory_probe/_medium_cells.py）"
_c = json.loads(CELLS.read_text(encoding="utf-8"))
assert _c["coordinate_source"] == "dataset_lab_bboxes_canvas_normalized"
assert _c["legacy_regression"]["exact_matches"] == _c["legacy_regression"]["v1_rows_seen"], \
    "legacy V1 regression 未通过 —— 拒绝启动"
assert (_c["n_g86"], _c["n_control"], _c["n_missed_non86"]) == (86, 1070, 164)
print(f"[ema-probe] cells OK: n={_c['n']} G86={_c['n_g86']} CONTROL={_c['n_control']} "
      f"regression={_c['legacy_regression']['exact_matches']}/{_c['legacy_regression']['v1_rows_seen']}")

print(f"[ema-probe] probe_target=trainer.ema.ema  probe_mode=eval  "
      f"coordinate_source={_c['coordinate_source']}")
for k in ("epochs", "batch", "imgsz", "seed", "lr0", "cos_lr", "close_mosaic", "ir_encoding",
          "box", "cls", "dfl", "patience"):
    print(f"    {k:<14} = {eff(k)}")

probe = EmaProbe(CELLS, HERE, observe_epochs=OBSERVE_0B)
model = YOLO(str(ROOT / "configs/yolo11m_sepstem.yaml"))

_gate = {"done": False}


def _attach(trainer):
    """on_pretrain_routine_end 在 trainer.py:317 触发，晚于 ema 创建（trainer.py:296）。"""
    from ultralytics.utils.torch_utils import de_parallel
    assert getattr(trainer, "ema", None) is not None, "ema 尚未创建 —— 时序假设失效"
    ema = de_parallel(trainer.ema.ema)
    probe.attach_model(ema)
    print("[ema-probe] attached to trainer.ema.ema", flush=True)

    # ---------- §13 in-process P4 gate ----------
    res = probe.self_check(ema, n_img=2)
    (HERE / "p4_gate_ema.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    for k, v in res.items():
        print(f"    P4 {k:<22} = {v}")
    if not res["PASS"]:
        _gate["failed"] = True
        raise RuntimeError(f"P4 non-intrusive gate FAILED: {res} —— 拒绝训练")
    _gate["done"] = True
    print("[ema-probe] P4 gate PASS", flush=True)


model.add_callback("on_pretrain_routine_end", _attach)
model.add_callback("on_fit_epoch_end", probe.on_fit_epoch_end)
model.add_callback("on_train_end", probe.on_train_end)
model.train(**kwargs)

# ---------- §16 收尾完整性门（任一 FAIL ⇒ STATUS=INVALID，由报告标注） ----------
assert _gate.get("done"), "P4 gate 未执行 —— 产物不可信"
assert probe.timing_log, "§6 观测时机证据缺失"
assert all(t["after_normal_ema_update"] for t in probe.timing_log), "§6 时机证据出现 False"
last_ep = probe.timing_log[-1]["epoch"]
assert last_ep == max(OBSERVE_0B), f"未跑到 0-based ep{max(OBSERVE_0B)}，实际停在 {last_ep}"
npzs = sorted(HERE.glob("epoch_*.npz"))
assert len(npzs) >= len(OBSERVE_0B), f"观测点缺失: {len(npzs)}/{len(OBSERVE_0B)}"

best = ROOT / kwargs.get("project", "runs") / RUN_NAME / "weights" / "best.pt"
assert best.exists(), f"final_eval 未产出 {best}"
import torch as _t
_ck = _t.load(best, map_location="cpu", weights_only=False)
stripped = (_ck.get("ema") is None and _ck.get("epoch") == -1)
print(f"[ema-probe] §16 收尾: epochs={last_ep+1}/300  npz={len(npzs)}  "
      f"best.pt={best.stat().st_size/1e6:.1f}MB  stripped={stripped}  "
      f"ema_key={_ck.get('ema')!r}")
assert stripped, "final_eval/strip_optimizer 未执行 —— 收尾不完整"
print(f"[ema-probe] done -> {HERE}")
