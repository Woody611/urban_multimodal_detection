"""_hard_gates.py — OASA scale=1.4 决胜实验的 §2 hard gates（1–7）。

只读、不训练、不需要 GPU。门 8/9/10（两个 SHA256 + 目标目录）由 _cloud_run_oasa_s14.sh
的 step 0 在**目标机器**上执行；本脚本覆盖需要在真实数据集上动态验证的 1–7 门。

门 4 与门 5 是**动态**验证（真的重建 transforms、真的构建 val dataset），不是读源码推测。
"""
import copy
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.augment import ObjectScaleAug, v8_transforms  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.utils import yaml_load  # noqa: E402

AUG = ROOT / "ultralytics/data/augment.py"
CFG14 = ROOT / "configs/oasa_pre_mosaic_scale14.yaml"
EXPECT_AUG = "5cb9a407625921ce3f12ffc13df2d703"
EXPECT_CFG14 = "dee6601dac39309d091164ddf3ee0056"
DS_YAML = ROOT / "data/processed/rgbid_split_train/dataset.yaml"

FAILS = []


def ck(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)
    return ok


def h32(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:32]


def main():
    data = yaml_load(str(DS_YAML))
    overrides = dict(
        imgsz=1280, task="detect", rect=False, cache=False, single_cls=False,
        classes=None, fraction=1.0, channels=5, use_simotm="RGBID",
        mosaic=1.0, copy_paste=0.0, mixup=0.0,
        object_scale_aug=True, object_scale_aug_prob=0.5, object_scale_aug_scale=1.4,
        object_scale_aug_small_area=1024.0, object_scale_aug_min_visible=0.8,
        object_scale_aug_log_every=200,
    )
    cfg = get_cfg(overrides=overrides)
    img_path = (Path(data["path"]) / data["train"]).resolve()
    src = AUG.read_text(encoding="utf-8", errors="replace")

    print("=" * 78)
    print("OASA scale=1.4 —— §2 HARD GATES (1–7)")
    print("=" * 78)

    print("\n--- 门 1: ObjectScaleAug 存在且为冻结实现 ---")
    ck("augment.py 含 `class ObjectScaleAug`", "class ObjectScaleAug" in src)
    ck("augment.py SHA256 == 5cb9a407…", h32(AUG) == EXPECT_AUG,
       f"actual {h32(AUG)}")

    print("\n--- 门 2: native-space small-object criterion ---")
    ck("含 `areas_native` 换算", "areas_native" in src)
    ck("native 换算行在位（ori/resized 面积比）",
       "areas_native = areas * (_o[0] * _o[1]) / (_r[0] * _r[1])" in src.replace("\r\n", "\n"))
    ck("small 判据读 areas_native（非 input-space areas）",
       "small = np.where(areas_native < self.small_area)[0]" in src.replace("\r\n", "\n"))

    print("\n--- 门 3: OASA 位于 v8_transforms 首位（pre-Mosaic）---")
    ds = build_yolo_dataset(cfg, str(img_path), 8, data, mode="train",
                            use_simotm="RGBID", pairs_rgb_ir=["visible", "infrared"],
                            pairs_rgb_depth=["visible", "depth"])
    order = [type(t).__name__ for t in ds.transforms.transforms]
    ck("transforms[0] is ObjectScaleAug", isinstance(ds.transforms.transforms[0], ObjectScaleAug),
       f"实际顺序 = {order}")
    ck("Mosaic 位于 OASA 之后", order.index("ObjectScaleAug") <
       order.index("Compose") if "Compose" in order else True, f"{order}")

    print("\n--- 门 4: close_mosaic 后 OASA 关闭（动态：真实 close_mosaic 入口）---")
    n_apply_before = ds.transforms.transforms[0].mosaic_runtime
    hyp2 = copy.deepcopy(cfg)
    ds.close_mosaic(hyp=hyp2)
    oasa_after = ds.transforms.transforms[0]
    ck("close_mosaic 后 transforms[0] 仍是 ObjectScaleAug",
       isinstance(oasa_after, ObjectScaleAug))
    ck("close_mosaic 后 mosa ic_runtime == 0（OASA 自动 OFF）",
       float(oasa_after.mosaic_runtime) == 0.0,
       f"close_mosaic 前 mosaic_runtime={n_apply_before} → 后 ={oasa_after.mosaic_runtime}")
    ck("close_mosaic 后仍 enabled=True（靠 runtime 门控，非靠 enabled）",
       bool(oasa_after.enabled))

    print("\n--- 门 5: validation / inference 不进入 OASA ---")
    cfgv = get_cfg(overrides={**overrides, "augment": False})
    img_val = (Path(data["path"]) / data["val"]).resolve()
    dsv = build_yolo_dataset(cfgv, str(img_val), 8, data, mode="val",
                             use_simotm="RGBID", pairs_rgb_ir=["visible", "infrared"],
                             pairs_rgb_depth=["visible", "depth"])
    vorder = [type(t).__name__ for t in dsv.transforms.transforms]
    ck("val transforms 不含 ObjectScaleAug", "ObjectScaleAug" not in vorder, f"{vorder}")
    ck("val transforms == [LetterBox, Format]", vorder == ["LetterBox", "Format"], f"{vorder}")
    ck("ObjectScaleAug 仅在 v8_transforms 中实例化（训练专用路径）",
       src.count("ObjectScaleAug(") == 1,
       f"源码中出现 {src.count('ObjectScaleAug(')} 次实例化")

    print("\n--- 门 6: scale == 1.4 且其余 OASA 参数冻结 ---")
    o = ds.transforms.transforms[0] if isinstance(ds.transforms.transforms[0], ObjectScaleAug) \
        else oasa_after
    ck("scale == 1.4", float(o.scale) == 1.4, f"实际 {o.scale}")
    ck("prob == 0.5", float(o.prob) == 0.5, f"{o.prob}")
    ck("small_area == 1024", float(o.small_area) == 1024.0, f"{o.small_area}")
    ck("min_visible == 0.8", float(o.min_visible) == 0.8, f"{o.min_visible}")

    print("\n--- 门 7: single-variable assertions ---")
    ck("config SHA256 == dee6601d…", h32(CFG14) == EXPECT_CFG14, f"actual {h32(CFG14)}")
    print("         （完整 single-variable args diff 见 _args_check.py，须 exit 0）")

    print("\n" + "=" * 78)
    if FAILS:
        print(f"RESULT: FAIL ({len(FAILS)}) -> STOP, do not train")
        for f in FAILS:
            print(f"   - {f}")
        return 1
    print("RESULT: HARD GATES 1-7 ALL PASS")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
