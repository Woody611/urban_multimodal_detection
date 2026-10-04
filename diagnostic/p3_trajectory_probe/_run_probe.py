"""_run_probe.py — Probe-P3-Trajectory 的**云端**训练入口（本机无 GPU，必须在云端执行）。

设计要点（为什么这样做）：
  1. 复用 `scripts/train.py` 的全部配置映射逻辑（`_load_yaml` / `_split_train_val_rgbid` /
     `_build_train_kwargs`），**不改 scripts/train.py 一个字节**；
     只在 `model.train(**kwargs)` 之前挂两个只读回调（`add_callback`）。
  2. **epochs 保持 300 不变**，用回调在 ep30 后置 `trainer.stop = True` 提前停机。
     理由：规格 §3 要求 ≤30 epoch，但 §2 要求 SCHEDULER 与 D′ 一致。若把 epochs 改成 30，
     cosine 会被拉紧（ep2 的 lr 因子 0.934 vs D′ 的 0.9934），违反 §2。
     保持 epochs=300 + 早停，可让**前 30 个 epoch 的 LR / 增强 / close_mosaic 相位与 D′ 逐位相同**。
  3. 只覆盖 `name`（= 输出目录名）。`name` 不参与训练数值（它由 experiment_name 派生，
     只决定 run 目录），本轮改它仅为避免与既有 run 目录冲突/覆盖（§1 禁止覆盖任何既有 artifact）。
     在 RectLate10 轮已实测确认 `name` 不是训练变量。

用法（云端，仓库根目录）：
  python -X utf8 scripts/preflight_check_train_config.py \
      --model_config configs/yolo11m_sepstem.yaml \
      --train_config configs/train_rgbid_sepstem_clahe.yaml --expect-ir-encoding clahe
  python -X utf8 diagnostic/p3_trajectory_probe/_integrity_gate.py          # 必须先 PASS
  python -X utf8 diagnostic/p3_trajectory_probe/_run_probe.py
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "diagnostic/p3_trajectory_probe"))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import de_parallel  # noqa: E402
from _probe import P3TrajProbe  # noqa: E402

OUT = ROOT / "diagnostic/p3_trajectory_probe"
OBS = (0, 1, 2, 5, 10, 15, 20, 25, 30)
STOP_AFTER = 30
RUN_NAME = "urban_multimodal_det_yolo11_rgbid_sepstem_clahe_PROBE30"

# 复用 scripts/train.py 的配置映射（不改该文件）
spec = importlib.util.spec_from_file_location("tr", ROOT / "scripts/train.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

TRAIN_CFG = ROOT / "configs/train_rgbid_sepstem_clahe.yaml"
MODEL_CFG = ROOT / "configs/yolo11m_sepstem.yaml"

train_cfg = tr._load_yaml(TRAIN_CFG)
dataset_cfg = tr._load_yaml(ROOT / "configs/dataset.yaml")

# ---- 与 scripts/train.py::main 完全相同的数据准备 ----
val_ratio = float(train_cfg.get("val_ratio", 0.2))
seed = int(train_cfg.get("seed", 42))
data_path = tr._split_train_val_rgbid(dataset_cfg, val_ratio, seed)
kwargs = tr._build_train_kwargs(train_cfg, data_path)

# ---- 规格一致性硬断言（§2）----
# 注意：`_build_train_kwargs` **不返回** resume / exist_ok（它们由 scripts/train.py:605 的 main() 设置），
# 也不返回 cls/dfl/rect/mosaic/close_mosaic/lrf（这些走 ultralytics/cfg/default.yaml 的默认值）。
# 因此这里显式补齐，并统一用「kwargs 优先、否则 default.yaml」的口径打印**有效值**。
_DEF = tr._load_yaml(ROOT / "ultralytics/cfg/default.yaml")
kwargs["resume"] = False          # 与 scripts/train.py:605 一致（无 resume_path）
kwargs["exist_ok"] = False        # 禁止覆盖：目标目录必须不存在
eff = lambda k: kwargs[k] if k in kwargs else _DEF.get(k)

assert kwargs["epochs"] == 300, f"epochs 必须保持 300（scheduler 一致性），实际 {kwargs['epochs']}"
assert kwargs["seed"] == 42 and kwargs["batch"] == 8 and kwargs["imgsz"] == 1280
assert kwargs["optimizer"] == "SGD" and kwargs["lr0"] == 0.005
assert kwargs["cos_lr"] is True and kwargs["resume"] is False
assert kwargs.get("cls_pw") is None, "cls_pw 必须为默认（无类别加权）"
assert eff("rect") is False, "rect 必须为 false"
assert eff("mosaic") == 1.0 and eff("close_mosaic") == 10, "增强相位必须与 D′ 一致"
assert eff("box") == 7.5 and eff("cls") == 0.5 and eff("dfl") == 1.5, "loss gain 必须与 D′ 一致"
assert kwargs["pretrained"] == "yolo11m.pt"

kwargs["name"] = RUN_NAME          # name 只决定输出目录，不参与训练数值
_tgt = ROOT / str(kwargs.get("project", "runs")) / RUN_NAME
assert not _tgt.exists(), f"目标目录已存在，拒绝覆盖：{_tgt}"

print("[run_probe] epochs=300 + stop_after=30（保持 D′ 的 LR/增强相位）")
print(f"[run_probe] name={kwargs['name']}  project={kwargs.get('project')}  target={_tgt}")
print("[run_probe] 有效值（kwargs 优先，否则 default.yaml）：")
for k in ("epochs", "batch", "imgsz", "seed", "optimizer", "lr0", "lrf", "momentum",
          "weight_decay", "warmup_epochs", "cos_lr", "pretrained", "resume", "rect",
          "mosaic", "close_mosaic", "scale", "translate", "ir_encoding", "channels",
          "use_simotm", "box", "cls", "dfl", "cls_pw", "amp", "workers"):
    print(f"    {k:<15} = {eff(k)}")

probe = P3TrajProbe(OUT / "medium_cells.json", OUT, observe_epochs=OBS, stop_after_epoch=STOP_AFTER)
model = YOLO(str(MODEL_CFG))


def _attach(trainer):                     # on_pretrain_routine_end：model/optimizer 已建好、训练未开始
    probe.attach_model(de_parallel(trainer.model))
    print(f"[run_probe] probe attached to {type(de_parallel(trainer.model)).__name__}", flush=True)


model.add_callback("on_pretrain_routine_end", _attach)
model.add_callback("on_fit_epoch_end", probe.on_fit_epoch_end)
model.add_callback("on_train_end", probe.on_train_end)
model.train(**kwargs)
print(f"[run_probe] done. artifacts in {OUT}")
