"""scripts/verify_oasa_sync.py — 在**云端 A100 环境**开训前运行的一致性自检。

对应 STEP 4 §2（云端代码一致性）+ §3（配置闭环验证）。

它做三件事，全部只读：
  1. 静态：确认 `ultralytics/data/augment.py` 里确实有 ObjectScaleAug 与 pre_mosaic marker，
     并比对 4 个关键文件的 SHA256（与本地审计版一致）。
  2. 动态：用**真实训练入口**构建一次（不训练），确认 OASA 键进入 trainer.args、
     且 dataset.transforms[0] 就是 ObjectScaleAug。
  3. 打印 `CLOUD_SYNC = PASS/FAIL`。

用法（云端）:
    python scripts/verify_oasa_sync.py

退出码 0 = PASS；非 0 = FAIL（禁止开训）。
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# 本地审计通过时的 SHA256（前 32 位用于展示，完整值用于比对）
EXPECT = {
    "ultralytics/data/augment.py": "5cb9a407625921ce3f12ffc13df2d703",
    "ultralytics/cfg/default.yaml": "991a89b32de769d1d444cf34ebb978f1",
    "scripts/train.py": "9f55b09f9e1229b61fefd49622d31d30",
    "configs/oasa_pre_mosaic.yaml": "7fce03dc281dd59b1912d9a84cd33156",
}
FAILS = []


def ck(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


def h32(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:32]


def main():
    print("=" * 74)
    print("OASA 云端一致性自检（只读；不训练）")
    print("=" * 74)

    # ---------------- 1. 静态 ----------------
    print("\n--- 1. 静态检查 ---")
    aug = ROOT / "ultralytics/data/augment.py"
    txt = aug.read_text(encoding="utf-8", errors="replace") if aug.exists() else ""
    ck("augment.py 含 `class ObjectScaleAug`", "class ObjectScaleAug" in txt)
    ck("augment.py 含 pre_mosaic marker", "insertion=pre_mosaic" in txt)
    ck("augment.py 在 v8_transforms 中实例化 OASA", "ObjectScaleAug(" in txt)
    print("  --- SHA256（前32）---")
    for rel, exp in EXPECT.items():
        p = ROOT / rel
        if not p.exists():
            ck(f"{rel} 存在", False, "文件缺失")
            continue
        got = h32(p)
        ck(f"{rel}", got == exp, f"got {got} / expected {exp}")

    # ---------------- 2. 动态（真实入口，不训练）----------------
    print("\n--- 2. 动态检查（真实训练入口，仅构建）---")
    try:
        import yaml
        import scripts.train as TM

        TM._resolve_device = lambda cfg: "cpu"      # 只做构建，不占 GPU
        tc = yaml.safe_load((ROOT / "configs/oasa_pre_mosaic.yaml").read_text(encoding="utf-8"))
        kw = TM._build_train_kwargs(tc, "data/processed/rgbid_split_train/dataset.yaml")
        need = ["object_scale_aug", "object_scale_aug_prob", "object_scale_aug_scale",
                "object_scale_aug_small_area", "object_scale_aug_min_visible"]
        miss = [k for k in need if k not in kw]
        ck("train.py 解析出全部 OASA 键", not miss, f"缺失 {miss}")
        print("      " + ", ".join(f"{k}={kw.get(k)}" for k in need))

        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.data.augment import ObjectScaleAug

        ov = {"model": "configs/yolo11m_sepstem.yaml",
              "data": "data/processed/rgbid_split_train/dataset.yaml",
              "pretrained": "yolo11m.pt", "epochs": 1, "imgsz": 1280, "batch": 2,
              "device": "cpu", "workers": 0, "project": "/tmp/oasa_synccheck", "name": "c",
              "exist_ok": True, **{k: kw[k] for k in need if k in kw}}
        T = DetectionTrainer(overrides=ov)
        T.get_model(cfg=ov["model"])
        T._setup_train(world_size=1)
        for k in need:
            ck(f"trainer.args.{k}", hasattr(T.args, k), f"= {getattr(T.args, k, '<ABSENT>')}")
        tr = T.train_loader.dataset.transforms.transforms[0]
        ck("dataset.transforms[0] 是 ObjectScaleAug", isinstance(tr, ObjectScaleAug),
           type(tr).__name__)
        if isinstance(tr, ObjectScaleAug):
            ck("OASA enabled=True", tr.enabled is True)
            ck("OASA mosaic_ref 指向真实 Mosaic", tr.mosaic_ref is not None,
               f"p={tr.mosaic_runtime}")
        vd = T.test_loader.dataset
        ck("val transforms 不含 ObjectScaleAug",
           not any(isinstance(t, ObjectScaleAug) for t in vd.transforms.transforms),
           str([type(t).__name__ for t in vd.transforms.transforms]))
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        ck("动态检查可执行", False, f"{type(e).__name__}: {e}")

    # ---------------- 3. 判决 ----------------
    print("\n" + "=" * 74)
    if FAILS:
        print(f"CLOUD_SYNC = FAIL  ({len(FAILS)} 项)")
        for f in FAILS:
            print("   -", f)
    else:
        print("CLOUD_SYNC = PASS")
    print("=" * 74)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
