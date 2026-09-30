"""_test_resume_remap_fix.py — 验证 `_transfer_rgb_pretrained` 的 resume 守卫修复。

三项断言：
  A. stock(yolo11m.pt) -> sepstem  ：**仍然**要 remap（不能被守卫误跳过），返回 ~550
  B. sepstem -> sepstem（即 resume）：返回 0，不抛错
  C. 惰性证明：B 之后模型 state_dict 与源 ckpt 的 state_dict **逐张量相同**
     —— 即守卫只跳过一个本会崩溃的 remap，不改变任何权重
"""
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from ultralytics.models.yolo.detect.train import _transfer_rgb_pretrained  # noqa: E402
from ultralytics.nn.tasks import (  # noqa: E402
    DetectionModel,
    attempt_load_one_weight,
    yaml_model_load,
)

SEPSTEM_CKPT = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_oasa/weights/best.pt"
STOCK = ROOT / "yolo11m.pt"

fails = []


def ck(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def build():
    # 必须用 yaml_model_load（不是 yaml_load）：scale 由**文件名**正则决定，
    # 传裸 dict 会让 scale 静默回落 'n'（stem out=16 / 511 键，而非 48 / 661）。
    # 这正是 memory 记录的 model-scale-gotcha。真实训练走 YOLO(yaml) 也是这条路径。
    cfg = yaml_model_load(str(ROOT / "configs/yolo11m_sepstem.yaml"))
    assert cfg.get("scale") == "m", f"scale 应为 'm'，实际 {cfg.get('scale')!r}"
    return DetectionModel(cfg, nc=12, verbose=False)


def main():
    print("=" * 74)
    print("resume remap 守卫 —— 回归测试")
    print("=" * 74)

    print("\n--- A. stock -> sepstem（正常首训路径，守卫必须**不**介入）---")
    m = build()
    src_stock, _ = attempt_load_one_weight(str(STOCK))
    m.load(src_stock)
    nA = _transfer_rgb_pretrained(m, src_stock)
    ck("A 仍执行 remap（未被守卫误跳过）", nA > 0, f"_transfer_rgb_pretrained 返回 {nA}")
    ck("A 迁移张量数符合预期(~550)", 500 < nA < 600, f"{nA}")

    print("\n--- B. sepstem -> sepstem（resume 路径，守卫必须介入）---")
    mb = build()
    ckpt = torch.load(str(SEPSTEM_CKPT), map_location="cpu", weights_only=False)
    src_sep = ckpt["model"]
    matched = mb.load(src_sep)
    n_match = len(matched) if matched is not None else -1
    print(f"       model.load() 匹配键数 = {n_match} / {len(mb.state_dict())}")
    try:
        nB = _transfer_rgb_pretrained(mb, src_sep)
        ck("B 不再抛 RuntimeError", True, f"返回 {nB}")
        ck("B 返回 0（完全跳过 remap）", nB == 0, f"{nB}")
    except RuntimeError as e:
        ck("B 不再抛 RuntimeError", False, f"{e}")
        nB = None

    print("\n--- C. 惰性证明：修复后加载的权重与 ckpt 逐张量相同 ---")
    sd_m = mb.state_dict()
    sd_s = src_sep.state_dict()
    same = diff = 0
    bad = []
    for k, v in sd_s.items():
        if k not in sd_m:
            bad.append(f"missing in model: {k}")
            continue
        if v.shape != sd_m[k].shape:
            bad.append(f"shape {k}")
            continue
        if torch.equal(v.float(), sd_m[k].float()):
            same += 1
        else:
            diff += 1
            bad.append(f"value differs: {k}")
    ck("C 源 ckpt 的每个张量都在模型中且逐位相同", diff == 0 and not bad,
       f"相同 {same} / 不同 {diff}" + (f"  {bad[:3]}" if bad else ""))
    ck("C 张量总数 == 模型键数", same == len(sd_m), f"{same} vs {len(sd_m)}")

    print("\n" + "=" * 74)
    if fails:
        print(f"RESULT: FAIL ({len(fails)})")
        for f in fails:
            print(f"   - {f}")
        return 1
    print("RESULT: ALL PASS —— 守卫只阻断了本会崩溃的 remap，不改变任何权重")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
