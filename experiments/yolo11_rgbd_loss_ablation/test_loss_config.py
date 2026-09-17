"""Loss 消融配置回归测试（不启动训练）。

验证链路:
    L0/L1/L2 train.yaml
        -> scripts/train.py._build_train_kwargs  (生成 kwargs)
        -> ultralytics.cfg.get_cfg(overrides=kwargs)  (等价 trainer.args = get_cfg(...))
        -> trainer.args.box / .cls / .dfl
        -> (loss.py 中 self.hyp = args, 消费 self.hyp.box/.cls/.dfl)

并验证旧 configs/train.yaml（不含 box/cls/dfl）不会新增 override，保持 ultralytics 默认值。

用法:
    python experiments/yolo11_rgbd_loss_ablation/test_loss_config.py
退出码: 全部断言通过返回 0，任一失败返回非 0。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = PROJECT_ROOT / "experiments" / "yolo11_rgbd_loss_ablation"

EXPECTED = {
    "l0_baseline": {"box": 7.5, "cls": 0.5, "dfl": 1.5},
    "l1_box10": {"box": 10.0, "cls": 0.5, "dfl": 1.5},
    "l2_dfl2": {"box": 7.5, "cls": 0.5, "dfl": 2.0},
}
CONFIG_FILES = {
    name: PROJECT_ROOT / "configs" / f"train_{name}.yaml"
    for name in EXPECTED
}
OLD_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"


def _load_train_module():
    """以文件方式加载 scripts/train.py，暴露 _build_train_kwargs / _load_yaml。"""
    path = PROJECT_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("train_entry", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert_close(actual, expected, label):
    assert abs(float(actual) - float(expected)) < 1e-9, (
        f"{label}: 期望 {expected}, 实际 {actual}"
    )


def main() -> int:
    train_mod = _load_train_module()
    _build_train_kwargs = train_mod._build_train_kwargs
    _load_yaml = train_mod._load_yaml

    from ultralytics.cfg import get_cfg  # noqa: E402

    failures = 0

    # ---- 1. 逐个验证 L0/L1/L2 的 kwargs 与 get_cfg 落点 ----
    for name, expected in EXPECTED.items():
        cfg = _load_yaml(CONFIG_FILES[name])
        kwargs = _build_train_kwargs(cfg, data_path=str(ABLATION_DIR / "__dummy__.yaml"))

        # (a) kwargs 直接包含三个 loss 键
        for k in ("box", "cls", "dfl"):
            if k not in kwargs:
                print(f"[FAIL] {name}: kwargs 缺少 {k}")
                failures += 1
            else:
                _assert_close(kwargs[k], expected[k], f"{name}.kwargs.{k}")

        # (b) 等价 trainer.args = get_cfg(overrides=kwargs) 的落点
        args = get_cfg(overrides=kwargs)
        for k in ("box", "cls", "dfl"):
            _assert_close(getattr(args, k), expected[k], f"{name}.args.{k}")

        print(f"[OK] {name}: kwargs/args box={kwargs['box']} cls={kwargs['cls']} "
              f"dfl={kwargs['dfl']} (期望 {expected['box']}/{expected['cls']}/{expected['dfl']})")

    # ---- 2. 旧 configs/train.yaml 无 box/cls/dfl -> 不应新增 override ----
    if OLD_CONFIG.exists():
        old_cfg = _load_yaml(OLD_CONFIG)
        old_kwargs = _build_train_kwargs(old_cfg, data_path=str(ABLATION_DIR / "__dummy__.yaml"))
        for k in ("box", "cls", "dfl"):
            if k in old_cfg:
                print(f"[SKIP] 旧 {OLD_CONFIG.name} 显式含 {k}，非默认场景，跳过兼容性断言")
            elif k in old_kwargs:
                print(f"[FAIL] 旧 {OLD_CONFIG.name} 未配置 {k}，但 kwargs 被新增了 {k}={old_kwargs[k]}")
                failures += 1
            else:
                print(f"[OK] 旧 {OLD_CONFIG.name}: 未配置 {k}，kwargs 不新增 override（沿用 ultralytics 默认）")
    else:
        print(f"[SKIP] 未找到 {OLD_CONFIG}，跳过旧实验兼容性断言")

    print("\n===== 汇总 =====")
    if failures:
        print(f"[FAIL] 共 {failures} 处断言失败")
        return 1
    print("[PASS] 全部通过：L0/L1/L2 的 box/cls/dfl 正确流入 get_cfg/args，旧配置不受影响")
    return 0


if __name__ == "__main__":
    sys.exit(main())
