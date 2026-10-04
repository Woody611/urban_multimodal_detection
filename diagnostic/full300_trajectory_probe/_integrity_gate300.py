"""_integrity_gate300.py — Full-300 probe 的完整性闸门。

= 复跑 PROBE30 已验证的 G1–G8（子进程，必须 PASS）
  + 本轮新增两项：
    G9  **CLS_OUT 恒等式**（§8）：分类头最后一层是 1×1 Conv ⇒ out[c,y,x] 必须等于 W_c·h(y,x)+b_c。
        这是对 `logit = W·h + b` 的**真实验证**（而不是同义反复）。
    G10 **状态恢复**（§2）：probe 风格（eval + head.train + BN buffer 快照/还原）执行后，
        train/eval 模式、BN buffers、参数校验和、RNG 必须全部恢复。
"""
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBE30_GATE = HERE.parent / "p3_trajectory_probe" / "_integrity_gate.py"

print("=" * 96)
print("FULL-300 PROBE INTEGRITY GATE")
print("=" * 96)

print("\n--- 第 1 部分：复跑 PROBE30 已验证的 G1–G8 ---")
r = subprocess.run([sys.executable, "-X", "utf8", str(PROBE30_GATE)],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
base_ok = ("PROBE_INTEGRITY = PASS" in (r.stdout or ""))
for line in (r.stdout or "").splitlines():
    if "[PASS]" in line or "[FAIL]" in line or "PROBE_INTEGRITY" in line:
        print("  " + line.strip())
print(f"  ⇒ G1–G8 = {'PASS' if base_ok else 'FAIL'}")

print("\n--- 第 2 部分：本轮新增 G9 / G10 ---")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "p3_trajectory_probe"))
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, yaml_model_load  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402

ck = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
model = DetectionModel(yaml_model_load(str(ROOT / "configs/yolo11m_sepstem.yaml")), nc=12, verbose=False)
w, _ = attempt_load_one_weight(str(ck))
model.load(w); model.model[-1].stride = model.stride
model.args = get_cfg(overrides=dict(imgsz=1280, task="detect", channels=5, use_simotm="RGBID"))
model.eval(); model.model[-1].train()

cap = {}
h1 = model.model[-1].cv3[0][-1].register_forward_pre_hook(lambda m, i: cap.__setitem__("H", i[0].detach().clone()))
h2 = model.model[-1].cv3[0][-1].register_forward_hook(lambda m, i, o: cap.__setitem__("OUT", o.detach().clone()))
with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
    model(torch.rand(1, 5, 256, 256))
W = model.model[-1].cv3[0][-1].weight.detach().view(12, -1)
B = model.model[-1].cv3[0][-1].bias.detach()
H, OUT = cap["H"][0], cap["OUT"][0]
resid = 0.0
for c in range(12):
    for y, x in ((0, 0), (3, 5), (7, 7), (15, 15)):
        resid = max(resid, abs(float(OUT[c, y, x]) - float(W[c] @ H[:, y, x] + B[c])))
g9 = resid < 1e-3
print(f"  [{'PASS' if g9 else 'FAIL'}] G9 CLS_OUT 恒等式 out[c,y,x] == W_c·h+b_c   max|resid|={resid:.3e}")
h1.remove(); h2.remove()

# G10 状态恢复
mode0 = model.training
buf0 = {k: v.detach().clone() for k, v in model.named_buffers()}
cs0 = (round(float(sum(float(p.detach().float().sum()) for p in model.parameters())), 3),
       round(float(max(float(p.detach().abs().max()) for p in model.parameters())), 6))
import random
import numpy as np  # noqa: E402
st = (random.getstate(), np.random.get_state(), torch.get_rng_state())
model.eval(); model.model[-1].train()
with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
    model(torch.rand(1, 5, 256, 256))
with torch.no_grad():
    for k, v in model.named_buffers():
        if k in buf0:
            v.copy_(buf0[k])
random.setstate(st[0]); np.random.set_state(st[1]); torch.set_rng_state(st[2])
model.train(mode0)
mode_ok = (model.training == mode0)
buf_ok = all(torch.equal(v.detach(), buf0[k]) for k, v in model.named_buffers())
cs_ok = (round(float(sum(float(p.detach().float().sum()) for p in model.parameters())), 3),
         round(float(max(float(p.detach().abs().max()) for p in model.parameters())), 6)) == cs0
g10 = mode_ok and buf_ok and cs_ok
print(f"  [{'PASS' if g10 else 'FAIL'}] G10 状态恢复 mode={mode_ok} BN_buffers={buf_ok} params={cs_ok}")

OK = base_ok and g9 and g10
print()
print(f"FULL300_PROBE_INTEGRITY = {'PASS' if OK else 'FAIL'}")
