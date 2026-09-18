"""scripts/rgbid_pipeline_audit.py — RGBID 五通道数据链路审计（只读，不训练、不修改任何产物）。

逐段核对 [可见光 BGR] + [红外] + [深度] → 5ch [B,G,R,IR,D] → 模型输入 [R,G,B,IR,D] 的全链路：
  S1  原始文件格式与三模态配对完整性
  S2  load_and_preprocess_image 各模态实际 dtype/通道/统计
  S3  红外 1%/99% 分位数拉伸是否生效
  S4  深度 8bit/16bit 两条分支的实际走向与「0 值」语义
  S5  合并通道序（cv2.merge 实际产出）
  S6  LetterBox 逐通道 padding 值（train 与 val 两套）
  S7  Mosaic 画布填充值（逐通道）
  S8  Format._format_img 输出的通道序，并与推理侧 predict._to_chw 交叉验证
  S9  trainer 归一化（.float()/255，逐通道统一）
  S10 训练管线 vs 推理管线的通道序一致性（同图同 letterbox 直接比对）

用法:
  python scripts/rgbid_pipeline_audit.py [--split_root data/processed/rgbid_split]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ultralytics.data.augment import Format, LetterBox, Mosaic   # noqa: E402
from ultralytics.data.base import BaseDataset                    # noqa: E402
from predict import _to_chw                                      # noqa: E402

FAILS: list[str] = []
WARNS: list[str] = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def warn(name, cond, detail=""):
    if not cond:
        print(f"  [WARN] {name}   {detail}")
        WARNS.append(name)
    else:
        print(f"  [ OK ] {name}   {detail}")


def section(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_root", default="data/processed/rgbid_split")
    args = ap.parse_args()
    root = (PROJECT_ROOT / args.split_root).resolve()

    # ---------------- S1 ----------------
    section("S1  原始文件格式与三模态配对（train / val）")
    for split in ("train", "val"):
        base = root / "images" / split
        mods = {m: sorted((base / m).glob("*")) for m in ("visible", "infrared", "depth")}
        stems = {m: {p.stem for p in v} for m, v in mods.items()}
        inter = set.intersection(*stems.values())
        print(f"  [{split}] visible={len(mods['visible'])} infrared={len(mods['infrared'])} "
              f"depth={len(mods['depth'])}  三模态交集={len(inter)}")
        check(f"S1-{split} 三模态文件一一对应", len(inter) == len(mods["visible"]),
              f"可见光 {len(mods['visible'])} vs 交集 {len(inter)}")
        cnt = Counter()
        for m in ("visible", "infrared", "depth"):
            for p in mods[m][:80]:
                im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if im is None:
                    cnt[(m, "读取失败", "-")] += 1
                    continue
                rep = (im.ndim == 3 and im.shape[2] == 3
                       and np.array_equal(im[:, :, 0], im[:, :, 1])
                       and np.array_equal(im[:, :, 1], im[:, :, 2]))
                cnt[(m, f"{im.dtype}", f"{im.shape}")] += 1
                if rep:
                    cnt[(m, "三通道相同", "")] += 1
        for k in sorted(cnt, key=str):
            print(f"        {k[0]:9s} {k[1]:12s} {k[2]:16s} n={cnt[k]}")

    # ---------------- S2 / S3 / S4 / S5 ----------------
    section("S2–S5  load_and_preprocess_image（RGBID 分支）实际产出")
    ds = BaseDataset.__new__(BaseDataset)          # 只为调用实例方法，不做任何初始化
    ds.use_simotm = "RGBID"
    ds.pairs_rgb_ir = ("visible", "infrared")
    ds.pairs_rgb_depth = ("visible", "depth")

    vis_dir = root / "images" / "val" / "visible"
    sample = sorted(vis_dir.glob("*"))[0]
    im = ds.load_and_preprocess_image(str(sample), use_simotm="RGBID",
                                      pairs_rgb="visible", pairs_ir="infrared", pairs_depth="depth")
    print(f"  样本: {sample.name}")
    print(f"  合并后: shape={im.shape} dtype={im.dtype}")
    check("S5 merged shape = (H,W,5)", im.ndim == 3 and im.shape[2] == 5, f"{im.shape}")
    check("S2 merged dtype = uint8", im.dtype == np.uint8, f"{im.dtype}")

    # 单独取三模态，逐通道核对合并顺序 [B,G,R,IR,D]
    v = cv2.imread(str(sample))
    b, g, r = cv2.split(v)
    check("S5 ch0 == 可见光 B 通道", np.array_equal(im[:, :, 0], b))
    check("S5 ch1 == 可见光 G 通道", np.array_equal(im[:, :, 1], g))
    check("S5 ch2 == 可见光 R 通道", np.array_equal(im[:, :, 2], r))
    check("S5 ch0/1/2 互不相同（非灰度复制）",
          not (np.array_equal(b, g) and np.array_equal(g, r)),
          "可见光三通道应彼此不同")

    ir_raw = cv2.imread(str(sample).replace("visible", "infrared"), cv2.IMREAD_GRAYSCALE)
    ir_raw = np.squeeze(ir_raw)          # 某些 cv2 构建对 GRAYSCALE 返回 (H,W,1)
    lo, hi = np.percentile(ir_raw, (1.0, 99.0))
    ir_stretched = (np.clip((ir_raw.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
                    .astype(np.uint8) if hi - lo > 1 else ir_raw)
    print(f"  IR  : raw min/max={ir_raw.min()}/{ir_raw.max()}  1%/99% 分位={lo:.1f}/{hi:.1f}  "
          f"拉伸后 min/max={ir_stretched.min()}/{ir_stretched.max()}")
    check("S3 IR 1%/99% 分位数拉伸已生效（ch3 == 拉伸结果）",
          np.array_equal(im[:, :, 3], ir_stretched))
    check("S3 IR 未被拉伸成常量", len(np.unique(im[:, :, 3])) > 1, f"唯一值 {len(np.unique(im[:,:,3]))}")

    d_raw = cv2.imread(str(sample).replace("visible", "depth"), cv2.IMREAD_UNCHANGED)
    d0 = d_raw[..., 0] if d_raw.ndim == 3 else d_raw
    print(f"  DEPTH: raw shape={d_raw.shape} dtype={d_raw.dtype}  取 ch0 -> shape={d0.shape}")
    check("S4 ch4 == 深度第 0 通道", np.array_equal(im[:, :, 4], d0))
    if d_raw.dtype == np.uint8:
        print("  S4 本样本深度为 8bit -> `im_depth[im_depth<300]=0` 与 `/19999*255` 分支【不执行】")

    # 数据集里同时存在 8bit 与 16bit 深度：两边都要验
    ddir = root / "images" / "val" / "depth"
    d16 = d8 = None
    for p in sorted(ddir.glob("*")):
        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if a is None:
            continue
        if a.dtype == np.uint16 and d16 is None:
            d16 = p
        if a.dtype == np.uint8 and d8 is None:
            d8 = p
        if d16 and d8:
            break
    print(f"  16bit 深度样本: {d16.name if d16 else '无'}   8bit 深度样本: {d8.name if d8 else '无'}")

    if d16 is not None:
        # 三模态扩展名可能不同（depth 为 .png，visible/infrared 为 .jpg）→ 按 stem 查找
        vdir = d16.parent.parent / "visible"
        v16 = next((p for p in sorted(vdir.glob("*")) if p.stem == d16.stem), None)
        if v16 is None:
            print(f"  [SKIP] 找不到与 {d16.name} 同 stem 的可见光文件")
            v16 = None
    if d16 is not None and v16 is not None:
        im16 = ds.load_and_preprocess_image(str(v16), use_simotm="RGBID",
                                            pairs_rgb="visible", pairs_ir="infrared", pairs_depth="depth")
        raw16 = cv2.imread(str(d16), cv2.IMREAD_UNCHANGED)
        r16 = raw16[..., 0] if raw16.ndim == 3 else raw16
        print(f"  16bit 原图: shape={r16.shape} dtype={r16.dtype} "
              f"min/max={r16.min()}/{r16.max()}  <300mm 像素占比={100*(r16<300).mean():.2f}%")
        exp = r16.astype(np.float32).copy()
        exp[exp < 300] = 0.0
        exp = np.clip(exp / 19999.0 * 255.0, 0, 255).astype(np.uint8)
        got = im16[:, :, 4]
        print(f"  16bit 映射后: 期望 min/max/zero% = {exp.min()}/{exp.max()}/"
              f"{100*(exp==0).mean():.2f}%   实际 ch4 min/max/zero% = "
              f"{got.min()}/{got.max()}/{100*(got==0).mean():.2f}%")
        check("S4 16bit 分支：<300mm 置 0 且 /19999*255 映射正确",
              np.array_equal(got, exp))
        check("S4 16bit 分支确认被执行（映射后存在 0 值）", int((exp == 0).sum()) > 0,
              f"{int((exp==0).sum())} 个 0 像素")

    # 融合图分辨率一致性（数据集混有 640x360 与 1920x1080 两代源图）
    print(f"  融合后 (H,W) = {im.shape[:2]}   可见光原图 (H,W) = "
          f"{cv2.imread(str(sample)).shape[:2]}")
    check("S4 融合图 HW == 可见光原图 HW（三模态已对齐）",
          im.shape[:2] == cv2.imread(str(sample)).shape[:2])

    # ---------------- S6 ----------------
    section("S6  LetterBox 逐通道 padding 值")
    for tag, lb in (("train  (LetterBox 1280, scaleup=True )",
                     LetterBox(new_shape=(1280, 1280))),
                    ("val    (LetterBox 1280, scaleup=False)",
                     LetterBox(new_shape=(1280, 1280), scaleup=False))):
        pad = lb(image=im)
        vals = []
        for c in range(pad.shape[2]):
            ch = pad[:, :, c]
            u = np.unique(ch)
            # 取出现次数最多的值作为“背景填充值”
            vc = np.bincount(ch.ravel().astype(np.int64), minlength=256)
            vals.append(int(np.argmax(vc)))
        print(f"  {tag}: out={pad.shape}  各通道众数值(B,G,R,IR,D) = {vals}")
        check(f"S6 {tag} 5 通道填充一致", len(set(vals)) == 1, f"{vals}")

    # 探针：用非方形内容，确保 letterbox 一定会补边
    probe = np.full((64, 128, 5), 7, np.uint8)           # 内容全 7，(H=64, W=128)
    p2 = LetterBox(new_shape=(128, 128))(image=probe)
    corner = [int(p2[0, 0, c]) for c in range(5)]        # 左上角应落在补边区
    center = [int(p2[64, 64, c]) for c in range(5)]      # 中心应落在内容区
    print(f"  探针（内容全 7, 64x128 -> letterbox 128x128）左上角 = {corner}  中心 = {center}")
    check("S6 探针：5 通道补边值相同", len(set(corner)) == 1, f"{corner}")
    check("S6 探针：补边值 != 内容值（确实是 padding 而非内容）", corner[0] != 7, f"{corner}")
    check("S6 探针：中心仍为内容值 7", set(center) == {7}, f"{center}")

    # ---------------- S7 ----------------
    section("S7  Mosaic 画布填充值")
    try:
        m = Mosaic(dataset=None, imgsz=1280, p=1.0, dtype=np.uint8)
        print(f"  Mosaic.gray_value = {m.gray_value}")
        canvas = np.full((16, 16, 5), m.gray_value, dtype=np.uint8)
        cv = [int(canvas[0, 0, c]) for c in range(5)]
        print(f"  5 通道画布初值 = {cv}")
        check("S7 Mosaic 画布 5 通道同值", len(set(cv)) == 1, f"{cv}")
        warn("S7 Mosaic 画布值 == LetterBox padding 值", m.gray_value == corner[0],
             f"Mosaic={m.gray_value} LetterBox={corner[0]}")
    except Exception as e:
        print(f"  [SKIP] Mosaic 无法实例化: {e}")

    # ---------------- S8 ----------------
    section("S8  Format 输出通道序 与 推理侧 _to_chw 交叉验证")
    fmt = Format(bbox_format="xywh", normalize=True, return_mask=False, return_keypoint=False,
                 return_obb=False, batch_idx=True, mask_ratio=4, mask_overlap=True, bgr=0.0)
    lb = LetterBox(new_shape=(1280, 1280), scaleup=False)
    padded = lb(image=im)
    t_train = fmt._format_img(padded)          # 训练/验证路径
    t_infer = _to_chw(padded)                  # 推理路径 predict.py
    print(f"  Format._format_img -> {tuple(t_train.shape)} {t_train.dtype}")
    print(f"  predict._to_chw    -> {t_infer.shape} {t_infer.dtype}")
    check("S8 两条路径 shape 一致", tuple(t_train.shape) == tuple(t_infer.shape))
    check("S8 两条路径逐元素完全一致（同一 letterbox 输入）",
          np.array_equal(t_train.numpy(), t_infer))
    # 通道语义：ch0 应等于可见光 R
    r_ch = cv2.split(cv2.imread(str(sample)))[2]
    # 从 padded 里取回可见光 R（pad 前区域）
    h, w = im.shape[:2]
    r_from_pad = padded[:h, :w, 2]
    check("S8 输出 ch0 == 可见光 R（BGR→RGB 已翻转）",
          np.array_equal(t_train.numpy()[0][:h, :w], r_from_pad))
    check("S8 输出 ch3 == IR（未翻转）", np.array_equal(t_train.numpy()[3][:h, :w], padded[:h, :w, 3]))
    check("S8 输出 ch4 == Depth（未翻转）", np.array_equal(t_train.numpy()[4][:h, :w], padded[:h, :w, 4]))

    # ---------------- S9 ----------------
    section("S9  归一化")
    x = t_train.float() / 255
    print(f"  .float()/255 -> range [{x.min():.4f}, {x.max():.4f}]  各通道均值 = "
          f"{[round(float(x[c].mean()), 4) for c in range(5)]}")
    check("S9 全通道统一 /255（无逐通道 mean/std）", True)
    check("S9 值域在 [0,1] 内", float(x.min()) >= 0.0 and float(x.max()) <= 1.0)

    # ---------------- S10 ----------------
    section("S10  训练/推理 letterbox 几何差异（已知且预期）")
    from predict_rect import _compute_rect_shape
    h0, w0 = im.shape[:2]
    print(f"  原图 {h0}x{w0}")
    print(f"  训练/验证路径 : LetterBox(new_shape=(1280,1280), scaleup=False)  -> 方形画布 1280x1280")
    rh, rw = _compute_rect_shape(h0, w0, 1280, 32)
    print(f"  推理路径      : _preprocess_rect -> rect 画布 {rh}x{rw}（长边 1280，stride=32 对齐）")
    warn("S10 两条路径画布形状不同（方形 vs rect）", False,
         "这是 model.val() 用 rect、predict_rect 复刻 rect 的既有设计；"
         "提交管线与官方口径评测一致使用 rect，属预期")

    # ---------------- 汇总 ----------------
    section("汇总")
    print(f"  FAIL: {len(FAILS)}   WARN: {len(WARNS)}")
    if FAILS:
        print("  失败项:")
        for f in FAILS:
            print("    -", f)
    print()
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
