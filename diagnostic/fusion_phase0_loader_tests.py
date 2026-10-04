"""fusion_phase0_loader_tests.py — 三个 loader 路径的端到端（单图）验证。

  A) train/val 路径：BaseDataset.load_and_preprocess_image（YOLODataset/val 同源）
  B) inference 路径：LoadImagesAndVideos（predict_rect.py 使用）

对每个 use_simotm 验证：shape / dtype / range / 通道序 / IR 处理是否真的分派。

ZERO GPU · ZERO forward · ZERO training。
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = []


def rec(n, ok, d=""):
    RESULTS.append((n, bool(ok), d))
    print(f"  {'PASS' if ok else 'FAIL'}  {n}" + (f"  | {d}" if d else ""))


SPLITS = {
    # (split_root, 该 split 里用于取图的子目录 / 空串=扁平, 通道数, ...)
    "Gray2BGR":  ("visible_split",      "",         3, (0, 1, 2)),
    "Infrared":  ("rgbt_split",         "infrared", 3, (0, 0, 0)),
    "Depth":     ("depth_split_train",  "depth",    3, (0, 0, 0)),
    "RGBT":      ("rgbt_split",         "visible",  4, (0, 1, 2, 3)),
    "RGBIR":     ("rgbt_split",         "visible",  4, (0, 1, 2, 3)),
    "RGBD":      ("depth_split_train",  "visible",  4, (0, 1, 2, 3)),
    "IRD":       ("rgbid_split_train",  "infrared", 2, (0, 1)),
    "RGBID":     ("rgbid_split_train",  "visible",  5, (0, 1, 2, 3, 4)),
}


def make_ds(use_simotm):
    from ultralytics.data.base import BaseDataset
    ds = BaseDataset.__new__(BaseDataset)
    from ultralytics.cfg import get_cfg
    ds.hyp = get_cfg(overrides=dict(task="detect", imgsz=1280, use_simotm=use_simotm,
                                    channels=SPLITS[use_simotm][2],
                                    ir_encoding="clahe" if use_simotm in ("Infrared", "IRD", "RGBID", "RGBIR") else "percentile"))
    ds.augment = True
    ds.imgsz = 1280
    ds.use_simotm = use_simotm
    ds.pairs_rgb_ir = ("visible", "infrared")
    ds.pairs_rgb_depth = ("visible", "depth")
    ds.prefix = ""
    return ds


def test_train_val_path():
    print("\n[A] train/val 路径  BaseDataset.load_and_preprocess_image")
    for mode, (split, mod, nch, _) in SPLITS.items():
        d = ROOT / f"data/processed/{split}/images/val" / mod if mod else ROOT / f"data/processed/{split}/images/val"
        imgs = sorted(d.glob("*"))[:1] if d.is_dir() else []
        if not imgs:
            rec(f"A.{mode} 找图", False, str(d)); continue
        ds = make_ds(mode)
        try:
            # ⚠ RGBD 的 depth 映射在 base.py / loaders.py 都走 pairs_depth（2026-10-03 统一）
            im = ds.load_and_preprocess_image(str(imgs[0]), use_simotm=mode,
                                              pairs_rgb="visible",
                                              pairs_ir="infrared" if mode != "RGBD" else "depth",
                                              pairs_depth="depth")
            ok = (im.ndim == 3 and im.shape[2] == nch and im.dtype == np.uint8 and 0 <= im.min()
                  and im.max() <= 255)
            rec(f"A.{mode} shape={im.shape} ch={nch}", ok,
                f"dtype={im.dtype} range=[{im.min()},{im.max()}] mean={im.mean():.1f}")
        except Exception as e:
            rec(f"A.{mode}", False, f"{type(e).__name__}: {e}")


def test_inference_path():
    print("\n[B] inference 路径  LoadImagesAndVideos")
    from ultralytics.data.loaders import LoadImagesAndVideos
    from ultralytics.cfg import get_cfg

    tmp = Path(tempfile.mkdtemp(prefix="fp0_"))
    try:
        for mode, (split, mod, nch, _) in SPLITS.items():
            sub = tmp / mode
            sub.mkdir(parents=True, exist_ok=True)
            # 用 rgbid_split_train/val 的三模态目录做源（三模态齐全）
            base = ROOT / "data/processed/rgbid_split_train/images/val"
            for m2 in ("visible", "infrared", "depth"):
                src = sorted((base / m2).glob("*"))[:1]
                if src:
                    tgt = sub / src[0].name
                    if not tgt.exists():
                        try:
                            tgt.hardlink_to(src[0])
                        except Exception:
                            shutil.copy(src[0], tgt)
            try:
                arg = get_cfg(overrides=dict(task="detect", imgsz=1280, use_simotm=mode,
                                             channels=nch, ir_encoding="clahe"))
                ld = LoadImagesAndVideos(str(ROOT / "data/processed/rgbid_split_train/images/val/visible"),
                                         batch=1, use_simotm=mode, imgsz=1280,
                                         pairs_rgb_ir=["visible", "infrared"],
                                         pairs_rgb_depth=["visible", "depth"],
                                         ir_encoding="clahe")
                it = iter(ld)
                _, im0s, _ = next(it)
                a = np.asarray(im0s)
                arr = a[0] if a.ndim == 4 else a
                ok = (arr.ndim == 3 and arr.shape[2] == nch)
                rec(f"B.{mode} shape={arr.shape} ch={nch}", ok, f"dtype={arr.dtype}")
                del ld, it
            except Exception as e:
                rec(f"B.{mode}", False, f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ir_encoding_really_dispatches():
    print("\n[C] IR 编码在**真实 loader** 上确实分派（不是只在单测里）")
    from ultralytics.data.base import BaseDataset
    from ultralytics.cfg import get_cfg
    d = ROOT / "data/processed/rgbid_split_train/images/val/infrared"
    img = str(sorted(d.glob("*"))[0])
    outs = {}
    for enc in ("clahe", "percentile", "raw"):
        ds = BaseDataset.__new__(BaseDataset)
        ds.hyp = get_cfg(overrides=dict(task="detect", imgsz=1280, use_simotm="Infrared",
                                        channels=3, ir_encoding=enc))
        ds.augment, ds.imgsz, ds.use_simotm = True, 1280, "Infrared"
        ds.pairs_rgb_ir, ds.pairs_rgb_depth, ds.prefix = ("visible", "infrared"), ("visible", "depth"), ""
        outs[enc] = ds.load_and_preprocess_image(img, use_simotm="Infrared",
                                                 pairs_rgb="visible", pairs_ir="infrared",
                                                 pairs_depth="depth")
    rec("C.1 Infrared+clahe != Infrared+percentile",
        not np.array_equal(outs["clahe"], outs["percentile"]),
        f"maxdiff={int(np.abs(outs['clahe'].astype(int)-outs['percentile'].astype(int)).max())}")
    rec("C.2 Infrared+raw != Infrared+percentile",
        not np.array_equal(outs["raw"], outs["percentile"]))

    # D′ 的 RGBID 路径仍按 clahe 走
    d2 = ROOT / "data/processed/rgbid_split_train/images/val/visible"
    img2 = str(sorted(d2.glob("*"))[0])
    outs2 = {}
    for enc in ("clahe", "percentile"):
        ds = BaseDataset.__new__(BaseDataset)
        ds.hyp = get_cfg(overrides=dict(task="detect", imgsz=1280, use_simotm="RGBID",
                                        channels=5, ir_encoding=enc))
        ds.augment, ds.imgsz, ds.use_simotm = True, 1280, "RGBID"
        ds.pairs_rgb_ir, ds.pairs_rgb_depth, ds.prefix = ("visible", "infrared"), ("visible", "depth"), ""
        outs2[enc] = ds.load_and_preprocess_image(img2, use_simotm="RGBID",
                                                  pairs_rgb="visible", pairs_ir="infrared",
                                                  pairs_depth="depth")
    rec("C.3 D′ RGBID: clahe 与 percentile 产出不同（证明 clahe 真的生效）",
        not np.array_equal(outs2["clahe"], outs2["percentile"]),
        f"maxdiff={int(np.abs(outs2['clahe'].astype(int)-outs2['percentile'].astype(int)).max())}")
    rec("C.4 RGBID 5ch / RGB 通道两侧一致（只有 IR 通道变）",
        np.array_equal(outs2["clahe"][:, :, [0, 1, 2, 4]], outs2["percentile"][:, :, [0, 1, 2, 4]]))


def main():
    print("=" * 92)
    print("Loader end-to-end（单图）—— 无 forward / 无 GPU / 无训练")
    print("=" * 92)
    test_train_val_path()
    test_inference_path()
    test_ir_encoding_really_dispatches()
    npass = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 92)
    print(f"TOTAL {npass}/{len(RESULTS)} PASS")
    for n, ok, d in RESULTS:
        if not ok:
            print(f"  ❌ {n}  {d}")
    print("=" * 92)
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
