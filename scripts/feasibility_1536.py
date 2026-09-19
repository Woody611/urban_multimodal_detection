"""scripts/feasibility_1536.py — 1536 训练显存/速度可行性探测（不训练、不写任何 checkpoint）。

目的：在正式开训前，实测 imgsz=1536 下不同 batch 的显存占用与吞吐，确定可用 batch。
**不做真实训练**：只跑固定步数的 forward+backward（真实模型 + 真实检测损失），随后 `optimizer.zero_grad()`，
不保存、不更新任何持久状态（权重仅被原地更新若干步，进程结束后丢弃）。

用法（云端 GPU）:
    python scripts/feasibility_1536.py --weights runs/urban_multimodal_det_yolo11_rgbird_ir_quicktest/weights/best.pt \
        --model_config configs/yolo11m_earlyfusion.yaml --imgsz 1536 --batches 6,5,4,3,2 --iters 6

本机 CPU 校验（小尺寸，仅验证脚本逻辑）:
    python scripts/feasibility_1536.py --imgsz 256 --batches 2 --iters 2 --device cpu --allow-cpu
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import DetectionModel  # noqa: E402


# --- torch 版本兼容（本机 2.4.1+cu 用 torch.amp.*；云端 2.1.2 只支持 torch.cuda.amp.*）---
def _autocast(enabled):
    """返回一个可用于 with 的 autocast 上下文，兼容 torch 2.1 / 2.4。"""
    try:
        return torch.amp.autocast("cuda", enabled=enabled)      # torch >= 2.3 新 API
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)          # torch <= 2.2


def _grad_scaler(enabled):
    if hasattr(torch.amp, "GradScaler"):                         # torch >= 2.3
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)            # torch <= 2.2



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, default=None, help="起始权重（如 1280 best.pt）；缺省则随机初始化")
    ap.add_argument("--model_config", type=str, default="configs/yolo11m_earlyfusion.yaml")
    ap.add_argument("--channels", type=int, default=5)
    ap.add_argument("--nc", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=1536)
    ap.add_argument("--batches", type=str, default="6,5,4,3,2", help="逗号分隔的 batch 候选")
    ap.add_argument("--iters", type=int, default=6, help="每个 batch 的 fwd+bwd 步数（前 2 步计入预热，不计时）")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--allow-cpu", action="store_true", help="允许在 CPU 上跑（仅供脚本自检）")
    return ap.parse_args()


def make_batch(bs, ch, imgsz, device, n_obj=12):
    """构造一个结构合法的最小检测 batch（真实 DetectionModel.loss 需要的字段）。"""
    img = torch.rand(bs, ch, imgsz, imgsz, device=device)
    n = n_obj * bs
    batch = {
        "img": img,
        "cls": torch.zeros(n, 1, device=device),
        "bboxes": torch.rand(n, 4, device=device) * 0.6 + 0.2,   # 归一化 xywh，落在图内
        "batch_idx": torch.arange(bs, device=device).repeat_interleave(n_obj).float(),
    }
    return batch


def probe(model, criterion, optimizer, scaler, bs, ch, imgsz, device, iters, use_amp):
    """返回 (ok, peak_alloc_GB, peak_reserved_GB, img_per_s, err)"""
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    model.train()
    t0 = None
    n_timed = 0
    try:
        for it in range(iters):
            batch = make_batch(bs, ch, imgsz, device)
            with _autocast(use_amp):
                preds = model(batch["img"])
                loss, _ = criterion(preds, batch)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if not torch.isfinite(loss):
                return False, 0, 0, 0, f"loss 非有限: {loss.item()}"
            if it == 1:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
            elif it == iters - 1:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                n_timed = iters - 2
        dt = time.time() - t0
        ips = (n_timed * bs) / dt if dt > 0 else 0.0
        if device.type == "cuda":
            pa = torch.cuda.max_memory_allocated() / 1e9
            pr = torch.cuda.max_memory_reserved() / 1e9
        else:
            pa = pr = 0.0
        return True, pa, pr, ips, ""
    except torch.cuda.OutOfMemoryError as e:
        return False, 0, 0, 0, f"OOM: {str(e)[:80]}"
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            return False, 0, 0, 0, f"OOM: {msg[:80]}"
        return False, 0, 0, 0, f"RuntimeError: {msg[:120]}"


def main():
    a = parse_args()
    if a.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(a.device)
    if device.type == "cpu" and not a.allow_cpu:
        print("!! 无 GPU。本脚本用于测量显存，必须在 GPU 上运行。加 --allow-cpu 仅做逻辑自检。")
        return 2

    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}  总显存 = {p.total_memory/1e9:.1f} GB  "
              f"torch {torch.__version__}  cuda {torch.version.cuda}")
    else:
        print(f"设备: CPU（逻辑自检模式，显存数据无意义）  torch {torch.__version__}")

    cfg = ROOT / a.model_config
    model = DetectionModel(str(cfg), ch=a.channels, nc=a.nc, verbose=False)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"模型: {cfg.name}  ch={a.channels} nc={a.nc}  参数={n_param:,}")
    if a.weights:
        from ultralytics.nn.tasks import attempt_load_one_weight
        src, _ = attempt_load_one_weight(str(ROOT / a.weights))
        model.load(src)
        print(f"起始权重: {a.weights}（已加载）")
    model = model.to(device)
    # 真实 trainer 会在 set_model_attributes() 里挂上 args；此处复刻，使 init_criterion 可用。
    # 用 ultralytics 默认超参（box=7.5/cls=0.5/dfl=1.5，与 baseline 一致）；
    # loss 权重不影响显存结论，探测只需走通同一条损失路径。
    if not hasattr(model, "args"):
        from ultralytics.cfg import get_cfg
        model.args = get_cfg()

    criterion = model.init_criterion()
    use_amp = device.type == "cuda"

    print(f"\nimgsz={a.imgsz}  AMP={use_amp}  iters/batch={a.iters}")
    print(f"{'batch':>6}{'结果':>10}{'peak_alloc(GB)':>16}{'peak_reserved(GB)':>19}{'img/s':>10}   备注")
    print("-" * 78)
    results = []
    for bs in [int(x) for x in a.batches.split(",")]:
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.937, nesterov=True)
        scaler = _grad_scaler(use_amp)
        ok, pa, pr, ips, err = probe(model, criterion, opt, scaler, bs, a.channels, a.imgsz,
                                     device, a.iters, use_amp)
        results.append((bs, ok, pa, pr, ips, err))
        print(f"{bs:>6}{'OK' if ok else 'FAIL':>10}{pa:>16.2f}{pr:>19.2f}{ips:>10.1f}   {err}")
        del opt, scaler
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    ok_bs = [r[0] for r in results if r[1]]
    print("-" * 78)
    if not ok_bs:
        print("!! 所有候选 batch 均失败 —— 1536 在现有显存下不可行。不要把结论解释为「需要改模型结构」。")
        return 1
    print(f"可用 batch: {sorted(ok_bs, reverse=True)}   最大可用 = {max(ok_bs)}")
    ref = 8
    est = int(ref / 1.44)
    print(f"参考: 1280 batch={ref} -> 按 1.44x 面积估算 1536 约 batch={est}；"
          f"实测最大可用 {max(ok_bs)} -> {'与估算一致' if max(ok_bs)==est else '与估算不同，以实测为准'}")
    print(f"建议 Phase A 使用 batch = {max(ok_bs)}（若需更稳可降一档）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
