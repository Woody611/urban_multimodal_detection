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

# ============================================================
# E1（L12–L17 Region Response Gain）预注册常量 —— 只读，不得就地修改
# ============================================================
# 这些值来自 reports/E1_EXPERIMENT_CONTRACT.md，是**预先定死**的。
# 任何一项对不上都说明「你以为在跑 E1，其实不是」，必须开训前拦下。
E1_LAYERS = (13, 15, 17)          # SepStem graph 0-based 层号（实测 512ch：13/15 为 C3k2，17 为 C2PSA）
E1_BASE_PARAMS = 20_061_972       # D′ baseline
E1_TOTAL_PARAMS = 20_077_332      # D′ + 3 × (512·3² + 512) = +15,360
E1_MODULE_COUNT = 3
E1_NEW_TENSORS = 6                # 每站点 e1.dw.{weight,bias}
E1_LOG_MARKER = "E1 region response gain"   # detect/train.py 的 remap 日志标记
E1_PRETRAINED = "yolo11m.pt"


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
    ap.add_argument("--expect-e1", choices=["true", "false"], default=None,
                    help="E1（L12–L17 Region Response Gain）硬性断言："
                         "'true' = 本次必须是**开启的** E1 实验（防止忘记把 e1_enabled 翻成 true "
                         "而静默训成第二遍 D′）；'false' = 本次必须确实是 D′ 等价（e1 关闭）。"
                         "**不给该参数则完全不检查 E1**，既有调用行为逐字不变。")
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

    # 6) E1 硬性断言 —— 仅在显式给出 --expect-e1 时启用（否则本段完全不执行）
    #    动机：E1 的开关在 model yaml 的顶层键 e1_enabled 上。忘记把它从 false 翻成 true，
    #    训练**不会报错**，只会静默产出第二遍 D′（11 小时白跑，且单变量对照失效）。
    #    同 2026-09-20 的 ir_encoding 事故是同一类：静默回落、看不出、代价极高。
    if args.expect_e1 is not None:
        want_on = args.expect_e1 == "true"
        _fails_before_e1 = len(FAILS)
        print(f"\n--- E1 硬性断言（--expect-e1 {args.expect_e1}）---")

        # 读 yaml 顶层键（不看注释，只看解析结果），并做**严格** bool 判定。
        # 语义要点：--expect-e1 false 下「键缺失」是**合法**的 —— 一份完全没有 E1 键的
        # model yaml（如 configs/yolo11m_sepstem.yaml）本来就被 parse_model 当作关闭，
        # 它就是合法的 D′ 等价配置。不能因为「没有这个键」而判 FAIL。
        mc_raw = yaml.safe_load(mc_path.open(encoding="utf-8")) or {}
        raw_flag = mc_raw.get("e1_enabled", None)
        if want_on:
            check("e1_enabled 严格为 True（bool，非字符串/非真值）",
                  raw_flag is True, f"{raw_flag!r} ({type(raw_flag).__name__})")
        else:
            check("e1_enabled 不是 True（缺失或显式 false）",
                  raw_flag is not True,
                  f"{raw_flag!r}" + ("" if raw_flag is None else f" ({type(raw_flag).__name__})"))

        # e1_layers / e1_kernel：**若声明**则必须严格等于预注册值；
        # 且 want_on 时不允许缺失（缺失就意味着 E1 根本没打开）。
        raw_layers = mc_raw.get("e1_layers", None)
        layers_present = isinstance(raw_layers, list)
        layers_match = layers_present and tuple(int(x) for x in raw_layers) == E1_LAYERS
        if want_on:
            check(f"e1_layers 严格等于预注册集合 {list(E1_LAYERS)}", layers_match,
                  f"{raw_layers!r}" + ("" if layers_match else "  <- 与 Experiment Contract 不符"))
        else:
            check(f"e1_layers 未声明或严格等于 {list(E1_LAYERS)}",
                  raw_layers is None or layers_match,
                  f"{raw_layers!r}")

        raw_k = mc_raw.get("e1_kernel", None)
        if want_on:
            check("e1_kernel == 3", raw_k == 3, repr(raw_k))
        else:
            check("e1_kernel 未声明或 == 3", raw_k is None or raw_k == 3, repr(raw_k))

        # 实际构建模型，用**图与张量**而不是配置文本说话
        try:
            from ultralytics.nn.modules.block import RegionResponseGain
            from ultralytics.nn.tasks import DetectionModel as _DM

            m_e1 = _DM(str(mc_path), verbose=False)
            n_e1_params = sum(p.numel() for p in m_e1.parameters())
            n_mod = sum(1 for _ in m_e1.modules() if isinstance(_, RegionResponseGain))
            e1_idx = [i for i, _l in enumerate(m_e1.model)
                      if isinstance(getattr(_l, "e1", None), RegionResponseGain)]
            n_keys = sum(1 for k in m_e1.state_dict() if ".e1." in k)
            tr = sum(p.numel() for p in m_e1.parameters() if p.requires_grad)
            print(f"    （E1 模型：参数 {n_e1_params:,}（可训练 {tr:,}），{len(m_e1.model)} 层，"
                  f"E1 模块 {n_mod} 个 @ {e1_idx}，E1 张量 {n_keys} 个）")

            if want_on:
                check(f"实际参数量 == {E1_TOTAL_PARAMS:,}", n_e1_params == E1_TOTAL_PARAMS,
                      f"{n_e1_params:,}" + ("" if n_e1_params == E1_TOTAL_PARAMS
                                            else f"  <- 期望 D′({E1_BASE_PARAMS:,}) + 15,360；"
                                                 f"若等于 {E1_BASE_PARAMS:,} 就是忘了翻开关"))
                check(f"E1 module 数量 == {E1_MODULE_COUNT}", n_mod == E1_MODULE_COUNT, str(n_mod))
                check(f"E1 module 位于层 {list(E1_LAYERS)}", tuple(e1_idx) == E1_LAYERS, str(e1_idx))
                check(f"E1 新增张量 == {E1_NEW_TENSORS}", n_keys == E1_NEW_TENSORS, str(n_keys))
                check("模型 graph 含 RegionResponseGain", n_mod == E1_MODULE_COUNT,
                      f"实测 RegionResponseGain 实例数 = {n_mod}")

                # 日志标记：先证源码里有该字符串，再（若权重在本地）真跑一次 remap 抓日志
                dt = (PROJECT_ROOT / "ultralytics/models/yolo/detect/train.py")
                src_ok = dt.exists() and E1_LOG_MARKER in dt.read_text(encoding="utf-8", errors="replace")
                check(f"detect/train.py 含日志标记 {E1_LOG_MARKER!r}", src_ok,
                      "" if src_ok else "源码缺失 -> 云上 console 里将看不到 E1 生效标记")

                pw = PROJECT_ROOT / E1_PRETRAINED
                if pw.exists():
                    import logging

                    from ultralytics.models.yolo.detect.train import _transfer_rgb_pretrained

                    class _Cap(logging.Handler):
                        def __init__(self):
                            super().__init__()
                            self.lines = []

                        def emit(self, record):
                            self.lines.append(record.getMessage())

                    cap = _Cap()
                    lg = logging.getLogger("ultralytics")
                    lg.addHandler(cap)
                    try:
                        _transfer_rgb_pretrained(m_e1, str(pw))
                    finally:
                        lg.removeHandler(cap)
                    hits = [ln for ln in cap.lines if E1_LOG_MARKER in ln]
                    hit6 = any(f"{E1_NEW_TENSORS} new tensors" in ln for ln in hits)
                    check(f"真实 remap 日志出现 {E1_LOG_MARKER!r} 且报 {E1_NEW_TENSORS} new tensors",
                          bool(hits) and hit6,
                          (hits[0][-120:] if hits else "日志中未出现该标记"))
                else:
                    print(f"  [WARN] 未找到 {E1_PRETRAINED}，跳过「真跑 remap 抓日志」"
                          f"（训练前会自动下载；源码标记已单独断言）")
            else:
                check(f"实际参数量 == {E1_BASE_PARAMS:,}（D′ 等价）",
                      n_e1_params == E1_BASE_PARAMS, f"{n_e1_params:,}")
                check("E1 module 数量 == 0", n_mod == 0, str(n_mod))
                check("E1 新增张量 == 0", n_keys == 0, str(n_keys))
            print(f"    （全模型参数总数 {n_e1_params:,}）")
        except Exception as e:  # noqa: BLE001
            check("E1 模型可构建", False, f"{type(e).__name__}: {e}")

        if len(FAILS) == _fails_before_e1:
            print("  >>> E1 PRE-FLIGHT PASS")
        else:
            print("  >>> E1 PRE-FLIGHT FAIL")
            print("  >>> 禁止启动训练：本次配置并非你意图中的 E1 实验（见上方 FAIL 项）")

    # 7) 输出目录是否已存在（ultralytics 会 +1 递增，导致产物落错地方）
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
