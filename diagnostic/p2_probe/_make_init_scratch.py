"""diagnostic/p2_probe/_make_init_scratch.py — 生成 Design A（from-scratch）的 P2 初始权重。

目的：让 P2 模型获得与正式 baseline **完全相同**的初始化处理。
正式 baseline 的 from-scratch 初始化链（实读）：
    yolo11m.pt --model.load()--> 642/649 --_transfer_rgb_pretrained()--> 5ch stem
    => 只有 7 个键随机：1 个 5ch stem（由 remap 补齐）+ 6 个 nc 相关的 cv3 末层
P2 模型若直接用 `pretrained: yolo11m.pt`，因 Detect 索引 23→26 位移，
    Detect 的 161 个键**全部随机**，比 baseline 少获得 **861,894** 个 COCO 预训练参数（cv2 框回归塔）。
本脚本用一次 checkpoint 手术补齐该差异：
    yolo11m.pt + _transfer_rgb_pretrained + Detect 尺度重映射(cv2/cv3-tower/dfl)
    => P2 的 Detect 得到与 baseline **同构**的预训练/随机划分；新增 P2 尺度分支随机。

⚠️ 只读 yolo11m.pt；输出写到 diagnostic/ 下；不触碰任何正式产物。
用法:  python diagnostic/p2_probe/_make_init_scratch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, intersect_dicts  # noqa: E402
from ultralytics.models.yolo.detect.train import _transfer_rgb_pretrained                # noqa: E402

MODEL_YAML = ROOT / "configs/yolo11m_rgbid_p2_probe.yaml"
BASE_YAML = ROOT / "configs/yolo11m_earlyfusion.yaml"
YOLO11M = ROOT / "yolo11m.pt"
OUT = Path(__file__).resolve().parent / "p2_probe_init_scratch.pt"
OLD_IDX, NEW_IDX = 23, 26


def remap_detect(sd_new, sd_old, old_idx=OLD_IDX, new_idx=NEW_IDX):
    """按【检测尺度】把 3 尺度 Detect 的 P3/P4/P5 重映射到 4 尺度 Detect 的 index 1/2/3。

    逐前缀拷贝 + 形状校验：cv2 全部分支、cv3 的 tower 部分（.0./.1.）形状可匹配，
    cv3 的 nc 相关末层（.2.weight/.2.bias, 80 vs 12）形状不符 -> 跳过（保持随机，
    与 baseline 的随机集**完全一致**）；dfl 为共享层。
    """
    n, skip = 0, []
    prefixes = [f"model.{old_idx}.cv2.{o}." for o in (0, 1, 2)] + \
               [f"model.{old_idx}.cv3.{o}." for o in (0, 1, 2)] + \
               [f"model.{old_idx}.dfl."]
    for pre in prefixes:
        # 目标前缀：cv2/cv3 的尺度索引 +1；dfl 保持
        if ".cv2." in pre or ".cv3." in pre:
            br, s = pre.split(".")[2], int(pre.split(".")[3])
            tgt = f"model.{new_idx}.{br}.{s + 1}."
        else:
            tgt = f"model.{new_idx}.dfl."
        for k, v in sd_old.items():
            if k.startswith(pre):
                kn = tgt + k[len(pre):]
                if kn in sd_new and sd_new[kn].shape == v.shape:
                    sd_new[kn].copy_(v)
                    n += 1
                else:
                    skip.append((kn, tuple(sd_new[kn].shape) if kn in sd_new else "缺", tuple(v.shape)))
    return n, skip


def main():
    src, _ = attempt_load_one_weight(str(YOLO11M))
    src_sd = src.float().state_dict()

    # 参照：baseline 从 yolo11m.pt 的划分（115 预训练 / 6 随机）
    ref = DetectionModel(str(BASE_YAML), ch=5, nc=12, verbose=False)
    ref.load(src)
    _transfer_rgb_pretrained(ref, src)
    ref_sd = ref.state_dict()
    ref_det = [k for k in ref_sd if k.startswith("model.23.")]
    ref_pre = [k for k in ref_det if k in intersect_dicts(src_sd, ref_sd)]
    ref_rnd = [k for k in ref_det if k not in ref_pre]
    print(f"参照 baseline: Detect 键 {len(ref_det)}  预训练 {len(ref_pre)}  随机 {len(ref_rnd)}")
    print(f"           随机键 = {[k.split('.',2)[-1] for k in ref_rnd]}")

    m = DetectionModel(str(MODEL_YAML), ch=5, nc=12, verbose=False)
    m.load(src)
    _transfer_rgb_pretrained(m, src)
    sd = m.state_dict()
    n, skip = remap_detect(sd, src_sd)
    print(f"\nDetect 尺度重映射: 成功 {n} 张量, 形状不符跳过 {len(skip)}")
    for s in skip:
        print(f"   skip {s[0]}  目标{s[1]} vs 源{s[2]}")

    # 复检 P2 的 Detect 划分是否与 baseline 同构
    det = [k for k in sd if k.startswith(f"model.{NEW_IDX}.")]
    pre = [k for k in det if k in intersect_dicts(src_sd, sd)]
    rnd = [k for k in det if k not in pre]
    print(f"\n修正后 P2: Detect 键 {len(det)}  按名预训练 {len(pre)}  随机 {len(rnd)}")
    print(f"           随机键后缀 = {sorted(set(k.split('.',3)[-1] for k in rnd))}")
    print(f"           新增 P2 尺度(cv2.0/cv3.0)键数 = "
          f"{len([k for k in det if '.cv2.0.' in k or '.cv3.0.' in k])}")

    chk = DetectionModel(str(MODEL_YAML), ch=5, nc=12, verbose=False)
    chk.load({"model": m})
    csd = chk.state_dict()
    print(f"\n自检: 从 yaml 重建 + 加载本权重 = {len(intersect_dicts(sd, csd))}/{len(csd)}")

    torch.save({"model": m, "train_args": {}}, str(OUT))
    print(f"已写出 {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
