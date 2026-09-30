"""scripts/preflight_check_train_config.py — 开训前硬性自检（在**云上**、训练前立即运行）。

动机（2026-09-20 真实事故）：
    某次 D' 训练在云上用了「experiment_name 取自 _clahe 配置、但配置里没有 ir_encoding 键」的组合。
    base.py:333 静默回落到 percentile，于是 "IR-CLAHE + Separate-Stem" 实际训成了
    "percentile + Separate-Stem" —— 两个变量同时改变，单变量对照失效，11 小时白跑。

    本地 dry-run 拦不住这一类错误，因为 dry-run 用的是**本地**配置，而云上跑的是**另一份**配置。
    所以必须有一个在**云上**、用**即将真正使用的那两份配置文件**、走**与正式训练完全相同的解析路径**
    的前置检查。

它做什么：
    用 scripts/train.py 的 _build_train_kwargs 解析训练配置（与正式训练同一条代码路径），
    然后把「实际会传给 ultralytics 的 kwargs」逐项打印并断言。**不看注释，只看解析结果。**

用法（训练命令之前）：
    python scripts/preflight_check_train_config.py \
        --model_config configs/yolo11m_sepstem.yaml \
        --train_config configs/train_rgbid_sepstem_clahe.yaml \
        --expect-ir-encoding clahe

退出码 0 = 可以开训；非 0 = 禁止开训。
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from scripts.train import _build_train_kwargs  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(f"{name}: {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config", required=True)
    ap.add_argument("--train_config", required=True)
    ap.add_argument("--expect-ir-encoding", default=None,
                    help="若给出，则断言 ir_encoding 必须等于该值")
    ap.add_argument("--expect-channel-count", type=int, default=5)
    ap.add_argument("--expect-scale", default="m")
    ap.add_argument("--dataset_config", default="configs/dataset.yaml")
    args = ap.parse_args()

    print("=" * 74)
    print("PREFLIGHT — train config 自检（云上 · 训练前 · 不看注释只看解析结果）")
    print("=" * 74)
    print(f"  model_config = {args.model_config}")
    print(f"  train_config = {args.train_config}")

    tc_path = PROJECT_ROOT / args.train_config
    mc_path = PROJECT_ROOT / args.model_config
    if not check("train_config 存在", tc_path.exists(), str(tc_path)):
        return 1
    if not check("model_config 存在", mc_path.exists(), str(mc_path)):
        return 1

    train_cfg = yaml.safe_load(tc_path.open(encoding="utf-8")) or {}

    # ---- 与正式训练完全相同的解析路径 ----
    kwargs = _build_train_kwargs(train_cfg, "data/processed/<placeholder>/dataset.yaml")

    print("\n--- 解析后的关键 kwargs（即将传给 YOLO.train）---")
    for k in ("data", "epochs", "patience", "batch", "imgsz", "optimizer", "lr0",
              "weight_decay", "warmup_epochs", "cos_lr", "seed", "close_mosaic",
              "use_simotm", "channels", "ir_encoding", "pairs_rgb_ir", "pairs_rgb_depth",
              "project", "name"):
        print(f"    {k:<16} = {kwargs.get(k, '<ABSENT>')}")

    print("\n--- 断言 ---")
    # 1) 模态与通道
    check("use_simotm == RGBID", str(kwargs.get("use_simotm")) == "RGBID",
          repr(kwargs.get("use_simotm")))
    check(f"channels == {args.expect_channel_count}",
          int(kwargs.get("channels", -1)) == args.expect_channel_count,
          repr(kwargs.get("channels")))

    # 2) ir_encoding 必须显式存在（本次事故的核心）
    ir = kwargs.get("ir_encoding", None)
    check("ir_encoding 已显式声明（未静默回落 percentile）", ir is not None,
          repr(ir) if ir is not None
          else "ABSENT -> base.py 会回落到 percentile，IR 预处理将成为隐藏变量")
    if args.expect_ir_encoding is not None:
        check(f"ir_encoding == {args.expect_ir_encoding!r}",
              str(ir) == args.expect_ir_encoding, repr(ir))

    # 3) 模型 scale / 参数量
    from ultralytics.nn.tasks import DetectionModel, guess_model_scale
    scale = guess_model_scale(str(mc_path))
    check(f"模型 scale == {args.expect_scale!r}", scale == args.expect_scale, repr(scale))
    try:
        m = DetectionModel(str(mc_path), verbose=False)
        n = sum(p.numel() for p in m.parameters())
        print(f"    （模型参数 {n:,}，{len(m.model)} 层，ch={m.yaml.get('ch')}，"
              f"nc={getattr(m.model[-1], 'nc', None)}）")
    except Exception as e:  # noqa: BLE001
        check("model 可构建", False, f"{type(e).__name__}: {e}")

    # 4) ultralytics 侧参数校验 —— 复刻 trainer 里的 get_cfg() -> check_dict_alignment()
    #    2026-09-21 事故：云上那份 ultralytics/cfg/default.yaml 没有 ir_encoding 键，
    #    get_cfg 直接 SyntaxError: "'ir_encoding' is not a valid YOLO argument."，训练起不来。
    #    本检查用**同一个函数**在开训前就把它拦下来。
    print("\n--- ultralytics 参数校验（复刻 trainer 的 get_cfg）---")
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT, check_dict_alignment

        _has_ir = "ir_encoding" in DEFAULT_CFG_DICT
        check("ultralytics/cfg/default.yaml 含 ir_encoding", _has_ir,
              repr(DEFAULT_CFG_DICT.get("ir_encoding")) if _has_ir
              else "缺失 -> get_cfg 会抛 SyntaxError，训练无法启动（云上 ultralytics 过期）")
        try:
            check_dict_alignment(DEFAULT_CFG_DICT, dict(kwargs))
            check("check_dict_alignment 接受本次全部 kwargs", True)
        except Exception as e:  # noqa: BLE001
            check("check_dict_alignment 接受本次全部 kwargs", False,
                  f"{type(e).__name__}: {str(e).splitlines()[0]}")
    except Exception as e:  # noqa: BLE001
        check("ultralytics 参数校验可执行", False, f"{type(e).__name__}: {e}")

    # 5) fork 完整性 —— `ultralytics/data/` 被 .gitignore 吞掉（`.gitignore:4:data/`），
    #    **git 永远同步不到**。若云上是 git clone/pull 来的，这些文件会是上游原版，
    #    RGBID 5 通道分支 / IR-CLAHE / sepstem remap 全部不存在。
    print("\n--- fork 完整性（data/ 被 gitignore，git 同步不到）---")
    FORK_MARKERS = [
        ("ultralytics/cfg/default.yaml", "ir_encoding", "自定义配置键"),
        ("ultralytics/data/base.py", "_merge_channels_rgbid", "RGBID 5ch 合并"),
        ("ultralytics/data/base.py", "createCLAHE", "训练侧 IR-CLAHE"),
        ("ultralytics/data/loaders.py", "createCLAHE", "评测侧 IR-CLAHE"),
        ("ultralytics/data/base.py", "ir_encoding", "base.py 读取 ir_encoding"),
        ("ultralytics/models/yolo/detect/train.py", "_transfer_rgb_pretrained", "预训练迁移"),
        ("ultralytics/models/yolo/detect/train.py", "_remap_separate_stem", "三-stem remap"),
        ("ultralytics/nn/modules/conv.py", "class SilenceChannel", "SilenceChannel 模块"),
    ]
    for rel, marker, why in FORK_MARKERS:
        p = PROJECT_ROOT / rel
        ok = p.exists() and marker in p.read_text(encoding="utf-8", errors="replace")
        hint = ("文件缺失" if not p.exists()
                else "未找到标记 —— 云上这份 fork 可能是上游原版，需重新同步整个 ultralytics/")
        check(f"{rel} 含 {marker!r}（{why}）", ok, "" if ok else hint)

    # 6) 输出目录是否已存在（ultralytics 会 +1 递增，导致产物落错地方）
    exp = train_cfg.get("experiment_name")
    run_dir = PROJECT_ROOT / "runs" / str(exp) if exp else None
    if run_dir is not None:
        exists = run_dir.exists()
        print(f"    输出目录 runs/{exp} {'已存在' if exists else '不存在'}")
        if exists:
            n = sum(1 for _ in run_dir.rglob("*") if _.is_file())
            print(f"  [WARN] 目录已存在（{n} 个文件）—— exist_ok=false 时产物会落到 "
                  f"runs/{exp}2，请确认是否要覆盖/改名")

    print("\n" + "=" * 74)
    if FAILS:
        print(f"RESULT: FAIL ({len(FAILS)} 项)")
        for f in FAILS:
            print("   -", f)
        print("PREFLIGHT_NOT_READY  → 禁止开训")
    else:
        print("RESULT: ALL CHECKS PASSED")
        print("PREFLIGHT_READY  → 可以开训；开训后请立刻确认 args.yaml 里 ir_encoding 的值")
    print("=" * 74)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
