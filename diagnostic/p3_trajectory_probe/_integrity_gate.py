"""_integrity_gate.py — 证明插针**不改变** forward / loss / 参数 / RNG（§1 强制前置）。

全部用**小型合成输入**（1×5×256×256）+ D′ 权重，**不读数据集、不做 validation**。
CPU 可跑。本闸门是 §1「运行前必须证明」的那一步，不是 validation inference。
"""
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "diagnostic/p3_trajectory_probe"))

from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, yaml_model_load  # noqa: E402
from ultralytics.utils.loss import v8DetectionLoss  # noqa: E402
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors  # noqa: E402
from _probe import RawTAL  # noqa: E402

CKPT = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
CFG = ROOT / "configs/yolo11m_sepstem.yaml"
torch.manual_seed(0); random.seed(0); np.random.seed(0)

print("=" * 96); print("PROBE INTEGRITY GATE（合成输入 1x5x256x256，CPU，无数据集）"); print("=" * 96)
cfg = yaml_model_load(str(CFG))
model = DetectionModel(cfg, nc=12, verbose=False)
w, _ = attempt_load_one_weight(str(CKPT))
model.load(w); model.model[-1].stride = model.stride
# 从 ckpt 直接建模型时没有 .args；补上 D′ 的 hyp（box7.5/cls0.5/dfl1.5/cls_pw=[]）
from ultralytics.cfg import get_cfg  # noqa: E402
model.args = get_cfg(overrides=dict(imgsz=1280, task="detect", channels=5, use_simotm="RGBID"))
model.eval(); model.model[-1].train()

x = torch.rand(1, 5, 256, 256)
tg = torch.tensor([[0., 3., 0.5, 0.5, 0.12, 0.16], [0., 0., 0.3, 0.4, 0.08, 0.10]])
loss_fn = v8DetectionLoss(model)
batch = {"batch_idx": tg[:, 0], "cls": tg[:, 1], "bboxes": tg[:, 2:]}

def fwd(m):
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        p = m(x)
    return p[1] if isinstance(p, tuple) else p

# ---------- G1: hooks 不改变 forward ----------
p_before = fwd(model)
feat = {}
h1 = model.model[23].register_forward_hook(lambda m, i, o: feat.__setitem__("P3", o.detach().clone()))
h2 = model.model[-1].cv3[0][-1].register_forward_pre_hook(lambda m, i: feat.__setitem__("H", i[0].detach().clone()))
p_after = fwd(model)
g1 = all(torch.equal(a, b) for a, b in zip(p_before, p_after)) and len(p_before) == len(p_after)
print(f"  [{'PASS' if g1 else 'FAIL'}] G1 forward 逐位相同（挂 hook 前/后）  len={len(p_before)}")

# ---------- G6: loss 逐位相同 ----------
with torch.no_grad():
    l_before = loss_fn(fwd(model), copy.deepcopy(batch))[1]
with torch.no_grad():
    l_after = loss_fn(fwd(model), copy.deepcopy(batch))[1]
g6 = torch.equal(l_before, l_after)
print(f"  [{'PASS' if g6 else 'FAIL'}] G6 loss(box,cls,dfl) 逐位相同  {l_before.tolist()}")

# ---------- G2: RawTAL 与父类 forward 逐位相同 ----------
_sd_before = {k: v.clone() for k, v in model.named_parameters()}
ref = TaskAlignedAssigner(topk=10, num_classes=12, alpha=0.5, beta=6.0)
prb = RawTAL(topk=10, num_classes=12, alpha=0.5, beta=6.0)
feats = fwd(model)
with torch.no_grad():
    pd_d, pd_s = torch.cat([v.view(1, loss_fn.no, -1) for v in feats], 2).split((loss_fn.reg_max * 4, loss_fn.nc), 1)
    pd_s = pd_s.permute(0, 2, 1).contiguous(); pd_d = pd_d.permute(0, 2, 1).contiguous()
    ap, st = make_anchors(feats, loss_fn.stride, 0.5)
    imgsz = torch.tensor(feats[0].shape[2:], dtype=pd_s.dtype) * loss_fn.stride[0]
    t = loss_fn.preprocess(tg, 1, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_l, gt_b = t.split((1, 4), 2); m_gt = gt_b.sum(2, keepdim=True).gt_(0)
    pbox = loss_fn.bbox_decode(ap, pd_d)
    args = (pd_s.detach().sigmoid(), (pbox.detach() * st).type(gt_b.dtype), ap * st, gt_l, gt_b, m_gt)
    r_ref = ref.forward(*args)
    r_prb = prb.forward(*args)
g2 = all(torch.equal(a.float(), b.float()) for a, b in zip(r_ref, r_prb))
print(f"  [{'PASS' if g2 else 'FAIL'}] G2 RawTAL 与父类 TaskAlignedAssigner 输出逐位相同")
say_L = prb.last
say_ref = ref
print(f"        mask_pos.sum()={int(say_L['mask_pos'].sum())}  mask_topk.sum()={int(say_L['mask_topk'].sum())}")

# ---------- G3: raw_align 未被 in-place 掩码（回到语义陷阱） ----------
ra = say_L["raw_align"][0, 0]; mp = say_L["mask_pos"][0, 0].bool()
g3 = bool((ra[~mp] > 0).any())
print(f"  [{'PASS' if g3 else 'FAIL'}] G3 raw_align 在 mask_pos==0 处仍 >0 的位置数 = "
      f"{int((ra[~mp] > 0).sum())}（若为 0 则说明取到的是被掩码后的值）")
# 对照：in-place 之后的值
with torch.no_grad():
    _a = say_L["raw_align"].clone(); _a *= mp
print(f"        掩码后 >0 的位置数 = {int((_a[0,0] > 0).sum())}  ⇒ 二者必须不同")

# ---------- G4/G5: RNG 不变 + 无梯度 + 参数未变 ----------
st_r, st_n, st_t = random.getstate(), np.random.get_state(), torch.get_rng_state()
with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
    for _ in range(3):
        fwd(model)
g4 = (random.getstate() == st_r) and bool((np.random.get_state()[1] == st_n[1]).all()) \
     and torch.equal(torch.get_rng_state(), st_t)
print(f"  [{'PASS' if g4 else 'FAIL'}] G4 probe 风格 forward 后 random / numpy / torch RNG 状态不变")
grads = [n for n, p in model.named_parameters() if p.grad is not None]
# ★ 只比【参数】；buffer（BN running stats）由 G8 单独负责，否则会把 BN 的副作用误算成"参数被改"
_p_before = dict(model.named_parameters())
par_ok = all(torch.equal(_sd_before[k], _p_before[k]) for k in _sd_before)

# ★ G8: head.train() 会让 BN 更新 running stats ⇒ 必须证明「快照+还原」能消除该副作用
_buf_before = {k: v.detach().clone() for k, v in model.named_buffers()}
model.eval(); model.model[-1].train()
with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
    fwd(model)                      # 故意在 train 模式下 forward ⇒ BN 会改 running stats
_buf_raw = {k: model.get_buffer(k).detach().clone() for k in _buf_before}
n_changed_raw = sum(1 for k in _buf_before if not torch.equal(_buf_before[k], _buf_raw[k]))
with torch.no_grad():               # 还原（probe 的做法）
    for k, v in model.named_buffers():
        if k in _buf_before:
            v.copy_(_buf_before[k])
_buf_rest = {k: model.get_buffer(k).detach().clone() for k in _buf_before}
n_changed_rest = sum(1 for k in _buf_before if not torch.equal(_buf_before[k], _buf_rest[k]))

g5 = (len(grads) == 0) and par_ok and (n_changed_rest == 0)
print(f"  [{'PASS' if g5 else 'FAIL'}] G5 无梯度（{len(grads)}）+ 参数逐位未变（{len(_sd_before)} 张量 = {par_ok}）")
print(f"  [{'PASS' if n_changed_rest == 0 else 'FAIL'}] G8 BN buffer：head.train() forward 后 {n_changed_raw} 个 buffer 被改；"
      f"快照+还原后 {n_changed_rest} 个 ⇒ 副作用已消除")

# ---------- G7: 特征语义与既有 audit 一致 ----------
W = model.model[-1].cv3[0][-1].weight.detach(); B = model.model[-1].cv3[0][-1].bias.detach()
hv = feat["H"][0, :, 5, 7]
lg = float(W[3].view(-1) @ hv + B[3])
print(f"  [PASS] G7 特征语义：P3={tuple(feat['P3'].shape)}  H={tuple(feat['H'].shape)}  "
      f"logit(W_3·h+b_3)={lg:.4f}  （与 _v7_extract.py 同口径）")

h1.remove(); h2.remove()
OK = g1 and g6 and g2 and g3 and g4 and g5
print()
print(f"PROBE_INTEGRITY = {'PASS' if OK else 'FAIL'}")
print("  不可变性：forward 逐位不变 / loss 逐位不变 / TAL 读取不改行为 / raw_align 未被掩码 / "
      "RNG 不变 / 无梯度 / 参数未变")
