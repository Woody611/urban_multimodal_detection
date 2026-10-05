# -*- coding: utf-8 -*-
"""§17 — SECOND-FIRE GATE（**M4 Seed-2 结束后**才运行；本轮不运行）。

不信任本次 P0/P1 的任何结论 —— 重跑全部关键门，并额外做**漂移检测**：
把当前文件 SHA 与 `audit_manifest.json`（审计时刻的基线）逐项对照。
任何一处漂移 ⇒ 必须停下来重新审，不得直接开训。

用法（M4 彻底结束后）：
    YOLO_OFFLINE=True python diagnostic/ir_aug_audit/second_fire_gate.py
退出码 0 = FINAL_PREFLIGHT PASS ⇒ 允许人工执行训练命令。
"""
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "audit_manifest.json")
RUN_DIR = os.path.join(ROOT, "runs", "urban_multimodal_det_yolo11_rgbid_sepstem_clahe_iraug")
M4_DIR = os.path.join(ROOT, "runs", "urban_multimodal_det_modality_m4_rgb_ir_seed2")

FAILS = []


def chk(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


def sha(p):
    if not os.path.exists(p):
        return "<MISSING>"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    print("=" * 74)
    print("SECOND-FIRE GATE — M4 Seed-2 结束后运行")
    print("=" * 74)

    man = json.load(open(MANIFEST, encoding="utf-8"))
    os.chdir(ROOT)

    print("\n--- 1) 源码/配置漂移检测（对照审计时刻 manifest）---")
    drifted = []
    for f, want in man["files"].items():
        got = sha(f)
        if got != want:
            drifted.append((f, want, got))
    chk("审计以来无文件漂移", not drifted,
        "; ".join(f"{f}: {w[:12]}->{g[:12]}" for f, w, g in drifted))

    print("\n--- 2) M4 Seed-2 必须已结束 ---")
    m4_files = []
    for r, _, fs in os.walk(M4_DIR):
        m4_files += [os.path.join(r, x) for x in fs]
    chk("M4 run 目录存在", os.path.isdir(M4_DIR), M4_DIR)
    print(f"      M4 dir files: {len(m4_files)}")
    print("      ⚠ 在**云上**确认：M4 进程已退出、GPU 显存已释放（本机无法观测远端进程）")
    chk("M4 配置 SHA 未变", sha("configs/train_modality_m4_rgb_ir_seed2.yaml")
        == man["files"]["configs/train_modality_m4_rgb_ir_seed2.yaml"])

    print("\n--- 3) 目标 run 目录必须是全新的 ---")
    chk("runs/…_iraug 不存在或为空", (not os.path.isdir(RUN_DIR))
        or (len(os.listdir(RUN_DIR)) == 0), RUN_DIR)

    print("\n--- 4) 重跑 resolved-config 单变量门 ---")
    r = subprocess.run([sys.executable, os.path.join(HERE, "gate_resolved_config.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    chk("SINGLE_VARIABLE_GATE (resolved)", r.returncode == 0, (r.stdout or "")[-160:])

    print("\n--- 5) 重跑 §6..§12 机器验证门 ---")
    r = subprocess.run([sys.executable, os.path.join(HERE, "launch_gate.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = [l for l in (r.stdout or "").splitlines() if "PASS" in l or "FAIL" in l or "====" in l]
    chk("§6..§12 launch gates", r.returncode == 0 and all("FAIL" not in l for l in tail),
        " | ".join(tail[-6:]))

    print("\n--- 6) 重跑旧码 vs 新码回归 A/B ---")
    r = subprocess.run([sys.executable, os.path.join(HERE, "regression_ab.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    chk("旧码-新码 IDENTICAL", "VERDICT: IDENTICAL" in (r.stdout or ""),
        [l for l in (r.stdout or "").splitlines() if "VERDICT" in l])

    print("\n--- 7) 训练前 preflight 脚本 ---")
    r = subprocess.run([sys.executable, "scripts/preflight_check_train_config.py",
                        "--model_config", "configs/yolo11m_sepstem.yaml",
                        "--train_config", "configs/train_rgbid_sepstem_clahe_iraug.yaml",
                        "--expect-ir-encoding", "clahe"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    chk("preflight exit code 0", r.returncode == 0, (r.stdout or "").strip().splitlines()[-1:])

    print("\n" + "=" * 74)
    if FAILS:
        print(f"FINAL_PREFLIGHT = FAIL   (blockers: {FAILS})")
        print("GPU_LAUNCH_AFTER_M4 = BLOCKED")
        return 1
    print("FINAL_PREFLIGHT = PASS")
    print("GPU_LAUNCH_AFTER_M4 = READY")
    print("\n★ 现在（且仅在现在）才允许**人工**执行：")
    print("    python scripts/train.py --model_config configs/yolo11m_sepstem.yaml \\")
    print("        --train_config configs/train_rgbid_sepstem_clahe_iraug.yaml")
    print("★ 训练启动后立刻（§18 30 秒硬门）：")
    print("    python diagnostic/ir_aug_audit/verify_run_args.py \\")
    print("        runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_iraug/args.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
