# `_before_snapshot/` — preflight 脚本的「改动前」快照（E1 preflight 改动的对照基线）

## 这里面是什么

`preflight_check_train_config.py` —— **2026-09-27 加入 E1 硬性断言之前**的副本。

## 为什么需要它

`scripts/preflight_check_train_config.py` 在本次改动前**已经是未提交的脏工作区状态**
（它本会话开始时就带 `ir_encoding` 相关改动，不在 git HEAD 里）。因此
`git show HEAD:scripts/preflight_check_train_config.py` **拿不到**改动前的版本 ——
这份快照是唯一留存的对照物。

## 用途：复现「baseline 路径零变化」的证明

```bash
# 改动前（用快照；必须在本目录的父目录为仓库根的层级下运行，
# 因为脚本用 Path(__file__).resolve().parents[1] 定位 PROJECT_ROOT）
python _before_snapshot/preflight_check_train_config.py \
    --model_config configs/yolo11m_sepstem.yaml \
    --train_config configs/train_rgbid_sepstem_clahe.yaml \
    --expect-ir-encoding clahe > /tmp/before.txt 2>&1

# 改动后（用正式脚本，且**不给** --expect-e1）
python scripts/preflight_check_train_config.py \
    --model_config configs/yolo11m_sepstem.yaml \
    --train_config configs/train_rgbid_sepstem_clahe.yaml \
    --expect-ir-encoding clahe > /tmp/after.txt 2>&1

diff /tmp/before.txt /tmp/after.txt   # 必须为空
```

E1 断言全部位于 `if args.expect_e1 is not None:` 之内，不给该参数时该段**完全不执行**，
故上面对照输出的差异应为空。已归档的实测输出见
`diagnostic/e1_region_gain/_preflight_baseline_BEFORE.txt` 与 `_preflight_baseline_AFTER.txt`。

## 附：`yolo11m_sepstem_e1_off.yaml`

`configs/yolo11m_sepstem_e1.yaml` 的一份 `e1_enabled: false` 变体，**仅用于演示**
「忘记翻开关」时 E1 preflight 会拦下（`diagnostic/e1_region_gain/_pf_should_fail.txt`）。
不参与任何训练。
