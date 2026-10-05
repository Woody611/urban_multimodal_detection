# -*- coding: utf-8 -*-
"""开训**之后立刻**跑的验证：确认云上真正生效的是 IR augmentation 实验，而不是悄悄变回 D′。

动机与 2026-09-20 的 `ir_encoding` 事故同构：配置注释/文件名写着 _iraug，但实际解析出的
args.yaml 里 IR aug 四个键全是默认恒等 ⇒ 训出来的是第二遍 D′，11 小时白跑。

用法（**训练刚启动、跑完第 1 个 epoch 之前**）：
    python diagnostic/ir_aug_audit/verify_run_args.py \
        runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_iraug/args.yaml

期望输出: IR_AUG_ACTIVE = YES，且 exit code 0。
"""
import sys
import yaml

EXPECT = {"ir_gamma": [0.75, 1.35], "ir_gamma_probability": 0.35,
          "ir_noise_std": 0.015, "ir_noise_probability": 0.20}


def main(argv):
    path = argv[1] if len(argv) > 1 else "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_iraug/args.yaml"
    a = yaml.safe_load(open(path, encoding="utf-8"))
    fails = []
    print(f"args.yaml = {path}")
    for k, v in EXPECT.items():
        got = a.get(k, "<ABSENT>")
        ok = (list(got) == list(v)) if isinstance(v, list) and isinstance(got, (list, tuple)) else (got == v)
        print(f"  [{'PASS' if ok else 'FAIL'}] {k} = {got}  (expect {v})")
        if not ok:
            fails.append(k)
    for k, v in (("ir_encoding", "clahe"), ("use_simotm", "RGBID"), ("channels", 5), ("imgsz", 1280),
                 ("epochs", 300), ("batch", 8), ("seed", 42), ("lr0", 0.005), ("patience", 0)):
        got = a.get(k, "<ABSENT>")
        ok = got == v
        print(f"  [{'PASS' if ok else 'FAIL'}] {k} = {got}  (expect {v})")
        if not ok:
            fails.append(k)
    if fails:
        print(f"\nIR_AUG_ACTIVE = NO  (BLOCKERS: {fails})")
        print("⇒ 立即停止本次训练：这不是 IR augmentation 实验。")
        return 1
    print("\nIR_AUG_ACTIVE = YES  —— 可以继续训练。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
