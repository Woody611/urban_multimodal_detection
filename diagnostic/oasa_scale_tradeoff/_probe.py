"""_probe.py — 确认能用真实链路构建 RGBID train dataset 并拿到 OASA 的输入契约。

只读探针：不训练、不改任何正式代码。目的仅是验证 harness 的数据通路。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.utils import yaml_load  # noqa: E402

DS_YAML = ROOT / "data/processed/rgbid_split_train/dataset.yaml"
data = yaml_load(str(DS_YAML))

overrides = dict(
    imgsz=1280, task="detect", rect=False, cache=False, single_cls=False,
    classes=None, fraction=1.0, channels=5, use_simotm="RGBID",
    mosaic=1.0, copy_paste=0.0, mixup=0.0,
    object_scale_aug=True, object_scale_aug_prob=0.5, object_scale_aug_scale=2.0,
    object_scale_aug_small_area=1024.0, object_scale_aug_min_visible=0.8,
    object_scale_aug_log_every=0,
)
cfg = get_cfg(overrides=overrides)
img_path = (Path(data["path"]) / data["train"]).resolve()
ds = build_yolo_dataset(
    cfg, str(img_path), 8, data, mode="train", use_simotm="RGBID",
    pairs_rgb_ir=["visible", "infrared"], pairs_rgb_depth=["visible", "depth"],
)

print("len(ds)              =", len(ds))
print("transforms           =", [type(t).__name__ for t in ds.transforms.transforms])
oasa = ds.transforms.transforms[0]
print("oasa.enabled         =", oasa.enabled)
print("oasa.mosaic_runtime  =", oasa.mosaic_runtime)
print("oasa params          =", dict(scale=oasa.scale, prob=oasa.prob,
                                    small_area=oasa.small_area,
                                    min_visible=oasa.min_visible))

lab = ds.get_image_and_label(0)
print()
print("img                  =", lab["img"].shape, lab["img"].dtype)
print("ori_shape            =", lab["ori_shape"], " resized_shape =", lab["resized_shape"])
print("cls                  =", lab["cls"].shape, lab["cls"].dtype)
print("bboxes               =", lab["instances"].bboxes.shape,
      lab["instances"]._bboxes.format, "normalized =", lab["instances"].normalized)
area_in = float(lab["ori_shape"][0] * lab["ori_shape"][1])
area_rs = float(lab["resized_shape"][0] * lab["resized_shape"][1])
print("native/input area r  =", area_in / area_rs)

# 全量统计：native small 图占比（用于与 STEP 2.5 的 25.1% 对照）
n_small_img = 0
n_gt = 0
n_small = 0
for i in range(len(ds)):
    l = ds.get_image_and_label(i)
    H, W = l["ori_shape"][:2]
    b = l["instances"].bboxes
    cls = l["cls"]
    if len(cls) == 0:
        continue
    bw = b[:, 2] * W
    bh = b[:, 3] * H
    a = np.clip(bw, 0, None) * np.clip(bh, 0, None)
    n_gt += len(a)
    s = int((a < 1024).sum())
    n_small += s
    if s > 0:
        n_small_img += 1

print()
print("total GT             =", n_gt)
print("native small GT      =", n_small, f"({n_small / n_gt:.1%})")
print("images w/ small GT   =", n_small_img, f"/ {len(ds)} = {n_small_img / len(ds):.1%}")
print()
print("STEP 2.5 参照：small-containing images = 25.1% ; small/GT = 12.0%")
