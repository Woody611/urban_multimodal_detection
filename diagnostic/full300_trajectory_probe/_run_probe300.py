"""_run_probe300.py — Full-300 Trajectory Probe 的**云端**训练入口（本机无 GPU）。

与 PROBE30 的唯一行为差异：
  · **不早停**（§3/§4）：不设 trainer.stop，跑满 300 epoch。
  · 观涙点扩到 19 个（含 50/75/…/275/289/299），§5。
  · 额外捕获 head_W / head_b 与 CLS_OUT 恒等式（§8/§9C/§10）。
其余（model / loss / TAL / data / aug / optimizer / scheduler / seed / batch / imgsz）与 D′ 逐项相同。

用法（云端，仓库根目录）：
  python -X utf8 scripts/preflight_check_train_config.py \
      --model_config configs/yolo11m_sepstem.yaml \
      --train_config configs/train_rgbid_sepstem_clahe.yaml --expect-ir-encoding clahe
  python -X utf8 diagnostic/full300_trajectory_probe/_integrity_gate300.py      # 必须先 PASS
  python -X utf8 diagnostic/full300_trajectory_probe/_dryrun300.py              # 无 GPU 干跑
  python -X utf8 diagnostic/full300_trajectory_probe/_run_probe300.py           # ~4.5–11 h
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "p3_trajectory_probe"))

from ultralytics import YOLO  # noqa: E402
from _probe300 import Probe300, OBSERVE_0B  # noqa: E402

CELLS = HERE.parent / "p3_trajectory_probe" / "medium_cells.json"
RUN_NAME = "urban_multimodal_det_yolo11_rgbid_sepstem_clahe_FULL300"

spec = importlib.util.spec_from_file_location("tr", ROOT / "scripts/train.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

TRAIN_CFG = ROOT / "configs/train_rgbid_sepstem_clahe.yaml"
MODEL_CFG = ROOT / "configs/yolo11m_sepstem.yaml"
train_cfg = tr._load_yaml(TRAIN_CFG)
dataset_cfg = tr._load_yaml(ROOT / "configs/dataset.yaml")
data_path = tr._split_train_val_rgbid(dataset_cfg, float(train_cfg.get("val_ratio", 0.2)),
                                      int(train_cfg.get("seed", 42)))
kwargs = tr._build_train_kwargs(train_cfg, data_path)
_DEF = tr._load_yaml(ROOT / "ultralytics/cfg/default.yaml")
eff = lambda k: kwargs[k] if k in kwargs else _DEF.get(k)
kwargs["resume"] = False
kwargs["exist_ok"] = False
kwargs["name"] = RUN_NAME

# ---- §1/§3 硬断言 ----
assert kwargs["epochs"] == 300, f"必须 300 epoch，实际 {kwargs['epochs']}"
assert kwargs["seed"] == 42 and kwargs["batch"] == 8 and kwargs["imgsz"] == 1280
assert kwargs["optimizer"] == "SGD" and kwargs["lr0"] == 0.005 and kwargs["cos_lr"] is True
assert kwargs["resume"] is False
assert kwargs.get("cls_pw") is None
assert eff("rect") is False and eff("mosaic") == 1.0 and eff("close_mosaic") == 10
assert eff("box") == 7.5 and eff("cls") == 0.5 and eff("dfl") == 1.5
assert kwargs.get("ir_encoding") == "clahe", f"ir_encoding={kwargs.get('ir_encoding')}"
assert kwargs["patience"] == 0
_tgt = ROOT / str(kwargs.get("project", "runs")) / RUN_NAME
assert not _tgt.exists(), f"目标目录已存在，拒绝覆盖：{_tgt}"

print(f"[run_probe300] epochs=300 跑满（**不早停**）  name={RUN_NAME}")
print(f"[run_probe300] 观测点（0-based）={list(OBSERVE_0B)}")
print(f"[run_probe300] 观测点（csv 口径）={[e + 1 for e in OBSERVE_0B]}")
for k in ("epochs", "batch", "imgsz", "seed", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
          "warmup_epochs", "cos_lr", "pretrained", "resume", "rect", "mosaic", "close_mosaic",
          "ir_encoding", "channels", "use_simotm", "box", "cls", "dfl", "cls_pw", "patience"):
    print(f"    {k:<15} = {eff(k)}")

probe = Probe300(CELLS, HERE, observe_epochs=OBSERVE_0B)
model = YOLO(str(MODEL_CFG))


def _attach(trainer):
    from ultralytics.utils.torch_utils import de_parallel
    probe.attach_model(de_parallel(trainer.model))
    print("[run_probe300] probe attached", flush=True)


model.add_callback("on_pretrain_routine_end", _attach)
model.add_callback("on_fit_epoch_end", probe.on_fit_epoch_end)
model.add_callback("on_train_end", probe.on_train_end)
model.train(**kwargs)
print(f"[run_probe300] done -> {HERE}")
