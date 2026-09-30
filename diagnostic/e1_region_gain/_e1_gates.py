"""diagnostic/e1_region_gain/_e1_gates.py — E1 生效审计（只读 / 不训练 / 不写 runs）

E1 = L12–L17 Region Response Gain（恒等初始化的 depthwise 局部响应增益）。

本脚本回答的**唯一**问题：E1 这个变量是否真的接进了
CONFIG -> CODE PATH -> MODEL GRAPH -> FORWARD -> LOSS -> OPTIMIZER -> CHECKPOINT。

它**不**回答「E1 是否有效」—— 那要训练后才知道。

门（任一 FAIL 即 STOP，不得修补后继续）：

  A 组  E1=false 的 baseline identity（与 D′ 逐位一致）
      G1  模型 yaml diff           仅 3 个 e1_* 键不同
      G2  state_dict key diff      空集
      G3  参数量 / 可训练参数量     20,061,972
      G4  module graph diff        31 层、类名序列与 save list 全同
      G5  forward 输出             逐位相同
      G6  loss                     逐位相同
      G7  gradient checksum        逐位相同

  B 组  E1=true 生效
      G8  模块定位                 恰在 13/15/17，RegionResponseGain，5120 params/站点
      G9  forward 调用计数         每次 forward 每模块恰好 1 次
      G10 gradient                 grad_norm(E1 params) > 0
      G11 optimizer 归属           走 trainer.build_optimizer 真实路径
      G12 checkpoint save/load     存载一致 + D′ 旧 ckpt 仍可构建（getattr 兼容）
      G16 共享权重零扰动            E1=true 与 D′ 的共享 key 逐位相同（不扰动 RNG 流）
      G17 预训练迁移               对 yolo11m.pt 真实跑 _transfer_rgb_pretrained

  C 组  §7 干预测试（强化版，已与用户确认替代字面 §7）
      SL1 扰动活性                dw.bias=0.1 -> target/下游/prediction/loss 全部确定性差异
      SL2 恒等精确性              不扰动时上述四项逐位等于 bypass
      SL3 局部性                  单 cell 扰动 -> 增益仅在 k//2 邻域内变化

用法：
    python -X utf8 -u diagnostic/e1_region_gain/_e1_gates.py

退出码 0 = EXPERIMENT READY（全部 PASS）；非 0 = 有 FAIL，禁止训练。
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.engine.trainer import BaseTrainer  # noqa: E402
from ultralytics.nn.modules.block import RegionResponseGain  # noqa: E402
from ultralytics.nn.tasks import DetectionModel, yaml_model_load  # noqa: E402
from scripts.train import _build_train_kwargs  # noqa: E402

# ---------------------------------------------------------------- 常量
BASELINE_YAML = "configs/yolo11m_sepstem.yaml"          # D′ baseline
E1_YAML = "configs/yolo11m_sepstem_e1.yaml"             # E1 开关载体
BASELINE_CKPT = "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
BASELINE_CKPT_SHA_PREFIX = "1cae45f7"                    # = V11/V13 的 1cae45f7…
PRETRAINED = "yolo11m.pt"
NC, CH = 12, 5
SEED = 42
IMG = 640            # CPU；逐位判据与分辨率无关
BATCH = 2
E1_LAYERS = [13, 15, 17]
E1_PARAMS_PER_SITE = 512 * 3 * 3 + 512                  # 5120
D_BASE_PARAMS = 20_061_972
E1_DELTA_PARAMS = 3 * E1_PARAMS_PER_SITE                # 15360

RESULTS: list[dict] = []


def gate(gid, name, ok, detail=""):
    RESULTS.append({"gate": gid, "name": name, "pass": bool(ok), "detail": str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {gid:<4} {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 构建
def load_cfg(yaml_path):
    """走与 YOLO(path) 完全相同的加载路径（scale 由**文件名**解析）。"""
    d = yaml_model_load(yaml_path)
    if d.get("scale") != "m":
        raise SystemExit(
            f"[FATAL] scale={d.get('scale')!r} for {yaml_path} —— 文件名未解析出 'm'，"
            f"会静默回退 nano（历史踩过的坑）。拒绝继续。"
        )
    return d


def build(cfg, seed=SEED):
    """用给定 cfg 构建模型。seed 固定 => 两臂的随机初始化逐位可比。"""
    torch.manual_seed(seed)
    return DetectionModel(deepcopy(cfg), ch=CH, nc=NC, verbose=False)


def fixed_batch(seed=0):
    """固定 batch：确定性输入 + 合成标注（落在真实归一化范围）。"""
    torch.manual_seed(seed)
    img = torch.rand(BATCH, CH, IMG, IMG)
    # 每图 3 个框：0.20/0.12/0.35 归一化边长，落在网格内可被 assigner 命中
    bboxes = torch.tensor([[0.50, 0.50, 0.20, 0.20],
                           [0.25, 0.30, 0.12, 0.12],
                           [0.70, 0.65, 0.35, 0.35]])
    batch_idx = torch.zeros(len(bboxes), dtype=torch.long)
    cls = torch.tensor([[0.0], [3.0], [7.0]])
    if BATCH > 1:
        bboxes = torch.cat([bboxes, bboxes], 0)
        batch_idx = torch.cat([batch_idx, torch.ones(len(bboxes) // 2, dtype=torch.long)], 0)
        cls = torch.cat([cls, cls], 0)
    return {"img": img, "bboxes": bboxes, "batch_idx": batch_idx, "cls": cls}


def forward_train(model, batch):
    """train 模式 forward（loss 走这条路）。"""
    model.train()
    return model(batch["img"])


_TRAIN_CFG = "configs/train_rgbid_sepstem_clahe_e1.yaml"
_DATA_YAML = "data/processed/rgbid_split_train/dataset.yaml"
_HYP_CACHE = {}


def attach_hyp(model):
    """复现 trainer 的 `self.model.args = self.args`（detect/train.py:392）。

    v8DetectionLoss 从 model.args 读 box/cls/dfl/cls_pw；裸 DetectionModel 没有这个属性，
    所以必须显式挂上。超参用与正式训练**同一条**映射函数解析，避免在 gate 里用上默认值。
    """
    if not _HYP_CACHE:
        import yaml as _yaml
        with open(_TRAIN_CFG, encoding="utf-8") as f:
            _HYP_CACHE["cfg"] = _yaml.safe_load(f)
    kw = _build_train_kwargs(_HYP_CACHE["cfg"], _DATA_YAML)
    model.args = get_cfg(overrides=kw)
    return model.args


def loss_and_grads(model, batch):
    """一次 forward+backward，返回 (loss 值, 全参数梯度快照)。"""
    attach_hyp(model)
    model.zero_grad(set_to_none=True)
    preds = forward_train(model, batch)
    crit = model.init_criterion()
    loss, items = crit(preds, batch)
    total = loss.sum()
    total.backward()
    grads = {n: (None if p.grad is None else p.grad.detach().clone())
             for n, p in model.named_parameters()}
    return float(total.detach()), grads


def grads_equal(g1, g2):
    """逐位比较梯度快照；返回 (ok, 首个差异描述)。"""
    if set(g1) != set(g2):
        return False, f"param set differs: {sorted(set(g1) ^ set(g2))[:3]}"
    for n in g1:
        a, b = g1[n], g2[n]
        if (a is None) != (b is None):
            return False, f"{n}: None mismatch"
        if a is not None and not torch.equal(a, b):
            return False, f"{n}: max|d|={float((a - b).abs().max()):.3e}"
    return True, ""


def main():
    print("=" * 78)
    print("E1 生效审计 — CONFIG -> CODE PATH -> GRAPH -> FORWARD -> LOSS -> OPTIMIZER")
    print("=" * 78)

    # ---------------- provenance ----------------
    print("\n[provenance]")
    prov = {}
    for p in (BASELINE_YAML, E1_YAML, "configs/train_rgbid_sepstem_clahe.yaml",
              "configs/train_rgbid_sepstem_clahe_e1.yaml",
              "ultralytics/nn/modules/block.py", "ultralytics/nn/tasks.py",
              "ultralytics/models/yolo/detect/train.py", PRETRAINED, BASELINE_CKPT):
        h = sha256_file(p)
        prov[p] = h
        print(f"  {h[:16]}  {p}")

    ckpt_sha = prov[BASELINE_CKPT]
    print(f"\n  baseline ckpt sha == {BASELINE_CKPT_SHA_PREFIX}… : "
          f"{ckpt_sha.startswith(BASELINE_CKPT_SHA_PREFIX)}")
    if not ckpt_sha.startswith(BASELINE_CKPT_SHA_PREFIX):
        gate("G0", "baseline checkpoint 未被触碰", False, ckpt_sha)
        return 1
    gate("G0", "baseline checkpoint 未被触碰", True, f"sha256={ckpt_sha[:16]}...")

    cfg_base = load_cfg(BASELINE_YAML)                 # D′
    # 两臂都由 gate **显式强制**，不依赖文件出厂值 —— 否则一旦把文件翻成 true（正式训练的
    # 执行状态），A 组的「E1=false」臂就会静默变成 true 而让 G2/G16 之类的比较失去意义。
    cfg_off = load_cfg(E1_YAML)
    cfg_off["e1_enabled"] = False
    cfg_on = deepcopy(cfg_off)
    cfg_on["e1_enabled"] = True
    shipped = load_cfg(E1_YAML).get("e1_enabled")
    print(f"\n[arm control] 文件出厂 e1_enabled={shipped!r}；"
          f"本审计两臂强制为 False / True（与出厂值无关）")

    # ================================================================
    print("\n[A 组] E1=false baseline identity gate")
    # ================================================================
    m_base = build(cfg_base)
    m_off = build(cfg_off)

    # ---- G1 yaml diff（排除 loader 注入的键：yaml_file 随文件名变，scale 必须相等）----
    LOADER_KEYS = {"yaml_file"}
    diff_keys = {k for k in set(cfg_base) | set(cfg_off)
                 if k not in LOADER_KEYS and cfg_base.get(k) != cfg_off.get(k)}
    gate("G1", "模型 yaml diff 仅 e1_* 键（排除 loader 注入）",
         diff_keys == {"e1_enabled", "e1_layers", "e1_kernel"} and cfg_base["scale"] == cfg_off["scale"] == "m",
         f"diff keys = {sorted(diff_keys)}; scale={cfg_base['scale']}/{cfg_off['scale']}")

    # ---- G2 state_dict key diff ----
    sd_b, sd_o = m_base.state_dict(), m_off.state_dict()
    key_diff = set(sd_b) ^ set(sd_o)
    gate("G2", "state_dict key diff 为空", not key_diff, f"diff = {sorted(key_diff)[:5]}")

    # ---- G3 参数量 / 可训练参数量 ----
    # 注意：ultralytics 的 DFL conv（1x16x1x1 = 16 个权重）在 Detect.__init__ 里被
    # requires_grad_(False)，故 trainable = total - 16 是**既有行为**，不是本实验引入的。
    p_b = sum(p.numel() for p in m_base.parameters())
    p_o = sum(p.numel() for p in m_off.parameters())
    t_b = sum(p.numel() for p in m_base.parameters() if p.requires_grad)
    t_o = sum(p.numel() for p in m_off.parameters() if p.requires_grad)
    frozen_b = sorted(n for n, p in m_base.named_parameters() if not p.requires_grad)
    frozen_o = sorted(n for n, p in m_off.named_parameters() if not p.requires_grad)
    gate("G3", "参数量 / 可训练参数量 / 冻结参数集 == D′",
         p_b == p_o == D_BASE_PARAMS and t_b == t_o and frozen_b == frozen_o,
         f"total D'={p_b} E1off={p_o}；trainable {t_b}/{t_o}；frozen={frozen_b}")

    # ---- G4 module graph ----
    cls_b = [type(x).__name__ for x in m_base.model]
    cls_o = [type(x).__name__ for x in m_off.model]
    gate("G4", "module graph 全同（类名序列 + save list）",
         cls_b == cls_o and m_base.save == m_off.save and len(cls_b) == len(cls_o) == 31,
         f"n={len(cls_o)} save_equal={m_base.save == m_off.save}")

    # ---- G5 forward 逐位 ----
    batch = fixed_batch()
    m_base.eval(); m_off.eval()
    with torch.no_grad():
        out_b = m_base(batch["img"])
        out_o = m_off(batch["img"])
    pred_b = out_b[0] if isinstance(out_b, (list, tuple)) else out_b
    pred_o = out_o[0] if isinstance(out_o, (list, tuple)) else out_o
    gate("G5", "forward 输出逐位相同", torch.equal(pred_b, pred_o),
         f"shape={tuple(pred_o.shape)} max|d|={float((pred_b - pred_o).abs().max()):.3e}")

    # ---- G6 / G7 loss + gradient 逐位 ----
    m_base2 = build(cfg_base); m_off2 = build(cfg_off)
    l_b, g_b = loss_and_grads(m_base2, batch)
    l_o, g_o = loss_and_grads(m_off2, batch)
    gate("G6", "loss 逐位相同", l_b == l_o, f"D'={l_b!r} E1off={l_o!r}")
    gok, gdet = grads_equal(g_b, g_o)
    gate("G7", "gradient 逐位相同", gok, gdet)

    # ================================================================
    print("\n[B 组] E1=true 生效 gate")
    # ================================================================
    m_on = build(cfg_on)

    # ---- G8 模块定位 ----
    found = []
    for i, lyr in enumerate(m_on.model):
        e1 = getattr(lyr, "e1", None)
        if isinstance(e1, RegionResponseGain):
            found.append(i)
    g8_ok = found == E1_LAYERS
    detail = f"located={found} expected={E1_LAYERS}"
    if g8_ok:
        per = {i: sum(p.numel() for p in m_on.model[i].e1.parameters()) for i in found}
        kinds = {i: type(m_on.model[i]).__name__ for i in found}
        g8_ok = all(v == E1_PARAMS_PER_SITE for v in per.values()) and kinds == {13: "C3k2", 15: "C3k2", 17: "C2PSA"}
        detail += f" params={per} host={kinds}"
    gate("G8", "E1 模块定位（层号 / 类 / 参数量）", g8_ok, detail)

    p_on = sum(p.numel() for p in m_on.parameters())
    gate("G8b", "E1=true 参数量 == D′ + 15,360", p_on == D_BASE_PARAMS + E1_DELTA_PARAMS,
         f"{p_on} (delta={p_on - D_BASE_PARAMS})")

    # ---- G16 共享权重零扰动（必须与 G8 同级严厉）----
    sd_on = m_on.state_dict()
    shared = [k for k in sd_on if ".e1." not in k]
    g16_ok, g16_detail = True, ""
    for k in shared:
        if k not in sd_b or not torch.equal(sd_on[k], sd_b[k]):
            g16_ok, g16_detail = False, f"first divergent shared key = {k}"
            break
    if g16_ok:
        g16_detail = f"{len(shared)} 个共享 key 逐位相同；新增 e1 key = {len(sd_on) - len(shared)}"
    gate("G16", "E1=true 未扰动共享权重初始化（RNG 流未变）", g16_ok, g16_detail)

    # ---- G9 forward 调用计数 ----
    counts = {i: 0 for i in E1_LAYERS}
    handles = []
    for i in E1_LAYERS:
        def mk(idx):
            def hook(mod, inp, out):
                counts[idx] += 1
            return hook
        handles.append(m_on.model[i].e1.register_forward_hook(mk(i)))
    m_on.eval()
    with torch.no_grad():
        out_on = m_on(batch["img"])
    for h in handles:
        h.remove()
    gate("G9", "forward 时 E1 每模块恰好执行 1 次",
         all(v == 1 for v in counts.values()), f"counts={counts}")

    # ---- SL2 恒等精确性（E1=true 在 t=0 逐位等于 bypass）----
    pred_on = out_on[0] if isinstance(out_on, (list, tuple)) else out_on
    gate("SL2", "E1=true 在 t=0 逐位等于 baseline（恒等初始化）",
         torch.equal(pred_on, pred_o),
         f"max|d|={float((pred_on - pred_o).abs().max()):.3e}")

    # ---- G10 / SL1 梯度 + 扰动活性 ----
    m_on2 = build(cfg_on)
    l_on, g_on = loss_and_grads(m_on2, batch)
    gate("G6b", "loss(E1=true) 在 t=0 逐位等于 baseline", l_on == l_o, f"{l_on!r} vs {l_o!r}")

    e1_param_names = [n for n, _ in m_on2.named_parameters() if ".e1." in n]
    gnorms = {}
    for n, p in m_on2.named_parameters():
        if ".e1." in n:
            gnorms[n] = 0.0 if p.grad is None else float(p.grad.norm())
    pos = {k: v for k, v in gnorms.items() if v > 0}
    gate("G10", "grad_norm(E1 params) > 0", len(pos) == len(gnorms) and len(gnorms) == 6,
         f"{len(pos)}/{len(gnorms)} positive; norm={ {k.split('.')[1] + '.' + k.split('.')[-1]: round(v, 6) for k, v in gnorms.items()} }")

    # SL1: 扰动 dw.bias = 0.1，检查 target / 下游 / prediction / loss 全部变化
    m_pert = build(cfg_on)
    feats = {}
    def cap(idx):
        def hook(mod, inp, out):
            feats[idx] = out.detach().clone()
        return hook
    hs = [m_pert.model[i].register_forward_hook(cap(i)) for i in (13, 15, 17, 20, 23)]
    m_pert.eval()
    with torch.no_grad():
        pert_base = m_pert(batch["img"])
    for h in hs:
        h.remove()
    base_feats = dict(feats)
    for i in E1_LAYERS:
        m_pert.model[i].e1.dw.bias.data.fill_(0.1)
    feats.clear()
    hs = [m_pert.model[i].register_forward_hook(cap(i)) for i in (13, 15, 17, 20, 23)]
    with torch.no_grad():
        pert_after = m_pert(batch["img"])
    for h in hs:
        h.remove()
    pb = pert_base[0] if isinstance(pert_base, (list, tuple)) else pert_base
    pa = pert_after[0] if isinstance(pert_after, (list, tuple)) else pert_after
    d_target = {i: float((feats[i] - base_feats[i]).abs().max()) for i in (13, 15, 17)}
    d_down = {i: float((feats[i] - base_feats[i]).abs().max()) for i in (20, 23)}
    d_pred = float((pa - pb).abs().max())
    # 更强判据：置 dw.bias=0.1 且 dw.weight=0 时增益是**逐点常数** 1+tanh(0.1)，故每个元素上
    # (y'-y)/y 是一个确定值。untrained 网络 eval 模式下激活很小（量级 1e-5），用绝对差会把
    # 真信号误判成噪声，用比值才是正确的量纲。
    #
    # ⚠ 期望值必须考虑**级联**：E1 三个位点在图上依次串行（13 -> 14 -> 15 -> 16 -> 17），
    #   所以第 k 个位点测得的是 (1+c)^k - 1，而不是各自独立的 c。实测与级联预测吻合到
    #   5 位有效数字（0.099668 / 0.209270 / 0.329795 vs 预测 0.099668 / 0.209268 / 0.329768）。
    c_exp = float(torch.tanh(torch.tensor(0.1)))
    order = sorted(E1_LAYERS)
    expected = {i: (1.0 + c_exp) ** (order.index(i) + 1) - 1.0 for i in order}
    ratios = {}
    for i in E1_LAYERS:
        b, d = base_feats[i], feats[i] - base_feats[i]
        msk = b.abs() > 1e-12
        ratios[i] = float((d[msk] / b[msk]).median()) if bool(msk.any()) else float("nan")
    # 容差 1e-3：级联路径上的 BN(eval) 是仿射而非线性（beta 项），C2PSA 的注意力亦非尺度等变，
    # 实测偏差 ~3e-5，1e-3 留足余量但仍能抓住任何量级错误。
    rel_ok = all(abs(ratios[i] - expected[i]) < 1e-3 for i in E1_LAYERS)
    m_pert.train()
    l_pert, _ = loss_and_grads(m_pert, batch)
    sl1_ok = (all(v > 0 for v in d_target.values()) and all(v > 0 for v in d_down.values())
              and d_pred > 0 and l_pert != l_b and rel_ok)
    gate("SL1", "扰动活性：级联增益逐点 == (1+tanh(0.1))^k-1，且 target/下游/pred/loss 全部差异", sl1_ok,
         f"measured={ {k: round(v, 6) for k, v in ratios.items()} } "
         f"expected={ {k: round(v, 6) for k, v in expected.items()} }；"
         f"|d| target={d_target} down={d_down} pred={d_pred:.3e} loss {l_b:.6f}->{l_pert:.6f}")

    # ---- SL3 局部性 ----
    torch.manual_seed(1)
    rg = RegionResponseGain(16, 3).eval()
    with torch.no_grad():
        rg.dw.weight.copy_(torch.randn_like(rg.dw.weight) * 0.1)
        x0 = torch.randn(1, 16, 9, 9)
        g0 = torch.tanh(rg.dw(x0))
        x1 = x0.clone(); x1[0, :, 4, 4] += 5.0     # 单 cell 扰动
        g1 = torch.tanh(rg.dw(x1))
        delta = (g1 - g0).abs().sum(1)[0]          # (9,9)
    nz = (delta > 0).nonzero()
    if len(nz):
        rows, cols = nz[:, 0], nz[:, 1]
        sl3_ok = bool(rows.min() >= 3 and rows.max() <= 5 and cols.min() >= 3 and cols.max() <= 5)
        sl3_detail = f"support rows {int(rows.min())}..{int(rows.max())} cols {int(cols.min())}..{int(cols.max())} (期望 3..5)"
    else:
        sl3_ok, sl3_detail = False, "gain 对单 cell 扰动完全无响应"
    gate("SL3", "局部性：增益只依赖 k//2 邻域（非全局池化）", sl3_ok, sl3_detail)

    # ---- G11 optimizer 归属（走 trainer.build_optimizer 真实路径）----
    class _Shim:  # name != "auto" 时 build_optimizer 不触碰 self，故裸对象即可
        pass
    opt = BaseTrainer.build_optimizer(_Shim(), m_on2, name="SGD", lr=5e-3, momentum=0.937, decay=5e-4)
    in_opt = {id(p) for grp in opt.param_groups for p in grp["params"]}
    e1_params = [p for n, p in m_on2.named_parameters() if ".e1." in n]
    n_in = sum(1 for p in e1_params if id(p) in in_opt)
    all_in = all(id(p) in in_opt for p in m_on2.parameters())
    gate("G11", "E1 参数 ∈ optimizer.param_groups（且全模型参数都在）",
         n_in == len(e1_params) and all_in,
         f"e1 {n_in}/{len(e1_params)}；groups={[len(g['params']) for g in opt.param_groups]}")

    # ---- G12 checkpoint save/load + 旧 ckpt 兼容 ----
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="e1_ckpt_"))
    ck = tmp / "e1.pt"
    torch.save({"model": deepcopy(m_on2).half()}, ck)
    reloaded = torch.load(ck, map_location="cpu", weights_only=False)["model"]
    r_found = [i for i, l in enumerate(reloaded.model) if isinstance(getattr(l, "e1", None), RegionResponseGain)]
    same_keys = set(reloaded.state_dict()) == set(m_on.state_dict())
    # D′ 旧 ckpt 在本代码下仍可构建并可 forward（getattr 兼容路径）
    legacy_ok, legacy_detail = True, ""
    try:
        legacy = torch.load(BASELINE_CKPT, map_location="cpu", weights_only=False)["model"]
        legacy = legacy.float().eval()   # ckpt 以 fp16 保存（ultralytics 惯例），输入用 fp32
        with torch.no_grad():
            legacy(torch.rand(1, CH, 320, 320))
        legacy_has_e1 = any(isinstance(getattr(l, "e1", None), RegionResponseGain) for l in legacy.model)
        legacy_ok = not legacy_has_e1
        legacy_detail = f"D′ ckpt forward OK；legacy 含 e1 = {legacy_has_e1}"
    except Exception as e:  # noqa: BLE001
        legacy_ok, legacy_detail = False, f"{type(e).__name__}: {e}"
    gate("G12", "checkpoint 存取一致 + D′ 旧 ckpt 兼容",
         r_found == E1_LAYERS and same_keys and legacy_ok,
         f"reload e1={r_found} keys_equal={same_keys} | {legacy_detail}")

    # ---- G12b 冻结推理路径端到端（YOLO(yaml).predict 与 YOLO(ckpt).predict 结构一致）----
    # 直接覆盖需求 #11：inference 加载 checkpoint 时模块结构一致。用真正的冻结推理入口，
    # 而不是只比 state_dict。输入按实测数据形态：5ch、1280 宽、rect。
    import numpy as np
    import yaml as _yaml
    from ultralytics import YOLO
    inf_ok, inf_detail = False, ""
    try:
        d_on = _yaml.safe_load(Path(E1_YAML).read_text(encoding="utf-8"))
        d_on["e1_enabled"] = True
        yml_on = tmp / "yolo11m_sepstem_e1_on.yaml"   # 文件名须含 'yolo11m' 才解析到 scale=m
        yml_on.write_text(_yaml.safe_dump(d_on, allow_unicode=True), encoding="utf-8")
        y1 = YOLO(str(yml_on))
        loc_y = [i for i, l in enumerate(y1.model.model)
                 if isinstance(getattr(l, "e1", None), RegionResponseGain)]
        p_on = sum(x.numel() for x in y1.model.parameters())
        ck_full = tmp / "e1_best.pt"
        torch.save({"model": y1.model.half(), "epoch": -1, "best_fitness": None, "date": "",
                    "version": "x", "train_args": {}, "ema": None, "updates": 0,
                    "optimizer": None}, ck_full)
        y2 = YOLO(str(ck_full))
        loc_c = [i for i, l in enumerate(y2.model.model)
                 if isinstance(getattr(l, "e1", None), RegionResponseGain)]
        img = np.random.randint(0, 255, (736, 1280, CH), dtype=np.uint8)   # 1280x736, 5ch
        res = y2.predict(img, imgsz=1280, conf=0.001, iou=0.7, max_det=300, rect=True, verbose=False)
        nbox = 0 if res[0].boxes is None else len(res[0].boxes)
        inf_ok = loc_y == loc_c == E1_LAYERS and p_on == D_BASE_PARAMS + E1_DELTA_PARAMS
        inf_detail = (f"YAML->predict E1@{loc_y} params={p_on}；ckpt->predict E1@{loc_c}；"
                      f"predict 输出 {nbox} 框")
    except Exception as e:  # noqa: BLE001
        inf_detail = f"{type(e).__name__}: {e}"
    gate("G12b", "冻结推理路径 YOLO(ckpt).predict 结构与训练一致", inf_ok, inf_detail)

    # ================================================================
    print("\n[B 组附加] G17 预训练迁移（真实 _transfer_rgb_pretrained）")
    # ================================================================
    from ultralytics.models.yolo.detect.train import _transfer_rgb_pretrained
    t_base = build(cfg_base)
    t_on = build(cfg_on)
    n_base = _transfer_rgb_pretrained(t_base, PRETRAINED)
    n_on = _transfer_rgb_pretrained(t_on, PRETRAINED)
    sdb, sdon = t_base.state_dict(), t_on.state_dict()
    shared_t = [k for k in sdon if ".e1." not in k]
    bad = [k for k in shared_t if k not in sdb or not torch.equal(sdon[k], sdb[k])]
    # n 只统计有预训练对应项的迁移；E1 的 6 个新张量走新例外、单独计数，故两臂 n 必须相等。
    # 「例外确实生效」由「本行没有 raise」直接证明 —— 没有该例外时这里会抛
    # no pretrained counterpart for 'model.13.e1.dw.weight'。
    gate("G17", "预训练迁移后共享 key 逐位一致 + E1 key 走新例外（未 raise）",
         n_base > 0 and n_on == n_base and not bad,
         f"transferred D'={n_base} E1={n_on}（E1 新增 6 个张量单独放行）；divergent={bad[:3]}")

    # ================================================================
    print("\n" + "=" * 78)
    n_fail = sum(1 for r in RESULTS if not r["pass"])
    print(f"总计 {len(RESULTS)} 门，FAIL {n_fail}")
    print("EXPERIMENT READY" if n_fail == 0 else "EXPERIMENT NOT READY")
    print("=" * 78)

    out = Path(__file__).with_name("_e1_gates.json")
    out.write_text(json.dumps({"provenance": prov, "gates": RESULTS,
                               "n_fail": n_fail,
                               "verdict": "READY" if n_fail == 0 else "NOT_READY"},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[written] {out}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
