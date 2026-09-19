"""diagnostic/p2_probe/_make_init.py — 生成 P2 Probe 的专用初始权重（一次性 checkpoint 手术）。

⚠️ 只读源权重：仅读取正式 best.pt，**绝不修改**它。输出写到 diagnostic/ 下。

做三件事：
  1. 从 configs/yolo11m_rgbid_p2_probe.yaml 建 4 尺度模型
  2. `load(best.pt)` —— 因 layer 0-22 索引与正式模型逐字相同，**528/528 键按名加载**
     （backbone + 原 FPN/PAN neck，18,681,240 参数）
  3. **Detect 尺度重映射**：正式 Detect 在 model.23（3 尺度 P3/P4/P5），
     P2 Probe 的 Detect 在 model.26（4 尺度 P2/P3/P4/P5），索引位移导致 0 按名匹配。
     按【检测尺度】把 old 的 index 0/1/2 映射到 new 的 index 1/2/3（cv2 与 cv3 逐分支），
     并复制共享的 dfl。实测 121 张量全部成功、0 形状不符。
  4. 保存 743/743 可加载的初始权重到 diagnostic/p2_probe/p2_probe_init.pt

用法:  python diagnostic/p2_probe/_make_init.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, intersect_dicts  # noqa: E402

MODEL_YAML = ROOT / "configs/yolo11m_rgbid_p2_probe.yaml"
BASE_CKPT = ROOT / "runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt"
OUT = Path(__file__).resolve().parent / "p2_probe_init.pt"
OLD_IDX, NEW_IDX = 23, 26


def remap_detect(sd_new, sd_old, old_idx=OLD_IDX, new_idx=NEW_IDX):
    """把 3 尺度 Detect 的 P3/P4/P5 按【检测尺度】重映射到 4 尺度 Detect 的 index 1/2/3。"""
    n, skip = 0, []
    for br in ("cv2", "cv3"):
        for old_s, new_s in ((0, 1), (1, 2), (2, 3)):        # old P3,P4,P5 -> new P3,P4,P5
            p_old, p_new = f"model.{old_idx}.{br}.{old_s}.", f"model.{new_idx}.{br}.{new_s}."
            for k, v in sd_old.items():
                if k.startswith(p_old):
                    kn = p_new + k[len(p_old):]
                    if kn in sd_new and sd_new[kn].shape == v.shape:
                        sd_new[kn].copy_(v)
                        n += 1
                    else:
                        skip.append((kn, tuple(sd_new[kn].shape) if kn in sd_new else "缺", tuple(v.shape)))
    for k, v in sd_old.items():                              # 共享层 dfl（不分尺度）
        if k.startswith(f"model.{old_idx}.dfl."):
            kn = f"model.{new_idx}.dfl." + k[len(f"model.{old_idx}.dfl."):]
            if kn in sd_new and sd_new[kn].shape == v.shape:
                sd_new[kn].copy_(v)
                n += 1
    return n, skip


def main():
    m = DetectionModel(str(MODEL_YAML), ch=5, nc=12, verbose=False)
    src, _ = attempt_load_one_weight(str(BASE_CKPT))
    src_sd = src.float().state_dict()

    m.load(src)                                             # 步骤 2：528/528
    sd = m.state_dict()
    n, skip = remap_detect(sd, src_sd)                      # 步骤 3
    print(f"Detect 尺度重映射：成功 {n} 张量，形状不符 {len(skip)}")

    inter = intersect_dicts(src_sd, sd)
    print(f"重映射后按名匹配 = {len(inter)} / {len(sd)}")

    # 决定性校验：从 yaml 重建并加载本 checkpoint，必须 743/743
    chk = DetectionModel(str(MODEL_YAML), ch=5, nc=12, verbose=False)
    chk.load({"model": m})
    csd = chk.state_dict()
    print(f"自检：从 yaml 重建 + 加载本权重 = "
          f"{len(intersect_dicts(sd, csd))}/{len(csd)}  100%={len(intersect_dicts(sd, csd)) == len(csd)}")

    torch.save({"model": m, "train_args": {}}, str(OUT))
    print(f"已写出 {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
