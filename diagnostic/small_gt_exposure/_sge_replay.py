"""_sge_replay.py — D′ small-GT effective-exposure Monte-Carlo replay（只读）

回答：一个 **native small GT**（native area < 1024 px²，沿用项目定义）在 D′ 的真实
train pipeline 里，一次 epoch 平均被实例化几次、多少次活到最终 target、最终多大。

与 V5 (`small_object_train_replay`) 的区别（这是本次的核心）：
    V5 按 **增强后** 的 GT 面积分层（它明确记录了身份不可追踪，244/250 样本失配），
    因此它只看得到「最终仍然小的 GT」。本脚本用 native→final 的 **逐 GT 身份**
    (见 `_sge_instrument.py`，其正确性已由 `_sge_verify.py` 实测 V1/V2/V3 PASS)
    回答「native 小的 GT 有多少活到了 final、被放大还是被缩小」。

真实 pipeline（实测自 runs/.../args.yaml）：
    OASA(off) → Mosaic(p=1.0) → CopyPaste(p=0) → RandomPerspective(translate .1, scale .5,
    degrees/shear/perspective 0) → MixUp(p=0) → Albumentations(p=0) → RandomHSV(5ch 无效)
    → FlipV(p=0) → FlipH(p=0.5) → Format
close_mosaic=10 ⇒ 最后 10/300 个 epoch 用 mosaic=0 重建 transforms（本脚本单独跑该 regime）。

RNG 声明：训练时的 RNG 流跨 dataloader worker，**无法复现**。本脚本用显式固定 seed 的
独立 Monte-Carlo 协议 ⇒ 输出是 **distribution 估计**，不是 exact count。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _sge_instrument as INSTR  # noqa: E402

from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data.base import BaseDataset  # noqa: E402
from ultralytics.data.build import build_yolo_dataset  # noqa: E402
from ultralytics.utils import yaml_load  # noqa: E402

OUT = Path(__file__).resolve().parent
DS_YAML = ROOT / "data/processed/rgbid_split_train/dataset.yaml"
BASE_SEED = 20260928

D_HYP = dict(imgsz=1280, task="detect", rect=False, cache=False, single_cls=False,
             classes=None, fraction=1.0, channels=5, use_simotm="RGBID",
             object_scale_aug=False, mixup=0.0, copy_paste=0.0,
             degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
             flipud=0.0, fliplr=0.5)

_STUB_LOAD = INSTR.make_stub_load_image(1)   # 1ch：几何等价（已实测）且快 13×



def _areas(inst):
    """按 instances 自身声明的 bbox 格式算面积。

    ⚠ 实测：mosaic 阶段（RP 输入）是 xyxy，而 final 阶段（Format 输入）是 **xywh**。
    首版一律按 xyxy 算 ⇒ `bb[:,2]-bb[:,0]` 算成 `w-cx`，负值被 clip 成 0，
    使所有 final 面积为 0、尺寸比全为 0.000。此处按 `_bboxes.format` 分支，避免再踩。
    """
    bb = np.asarray(inst.bboxes)
    if len(bb) == 0:
        return np.zeros(0)
    if str(inst._bboxes.format) == "xyxy":
        w = bb[:, 2] - bb[:, 0]
        h = bb[:, 3] - bb[:, 1]
    else:  # xywh
        w = bb[:, 2]
        h = bb[:, 3]
    return np.clip(w, 0, None) * np.clip(h, 0, None)


class State:
    """单个 sample 执行期间的回调状态。"""

    def __init__(self, agg):
        self.agg = agg
        self.n_native = 0
        self.cur = None
        self.ep_inst = set()      # 本 epoch 被实例化过的 uid
        self.ep_final = set()     # 本 epoch 至少产生一个 final target 的 uid

    def __call__(self, name, labels, uid):
        inst = labels.get("instances")
        if inst is None or uid is None:
            return
        uid = np.asarray(uid)
        if name == "native":
            is_main = self.n_native == 0
            self.n_native += 1
            for u in uid:
                u = int(u)
                self.agg["inst"][u] += 1
                (self.agg["inst_main"] if is_main else self.agg["inst_tile"])[u] += 1
                self.ep_inst.add(u)
        elif name == "mosaic":
            ar = _areas(inst)              # mosaic 画布(2560)空间
            for j, u in enumerate(uid):
                u = int(u)
                self.agg["mosaic"][u] += 1
                self.agg["MA_uid"].append(u)
                self.agg["MA_val"].append(float(ar[j]))
        elif name == "final":
            ar = _areas(inst)              # Format 之前：final 像素空间(1280)
            for j, u in enumerate(uid):
                u = int(u)
                self.agg["final"][u] += 1
                self.ep_final.add(u)
                self.agg["FA_uid"].append(u)
                self.agg["FA_val"].append(float(ar[j]))


def build(mosaic_p):
    data = yaml_load(str(DS_YAML))
    cfg = get_cfg(overrides=dict(D_HYP, mosaic=mosaic_p))
    img_path = (Path(data["path"]) / data["train"])
    return build_yolo_dataset(cfg, str(img_path), 8, data, mode="train", use_simotm="RGBID",
                              pairs_rgb_ir=["visible", "infrared"], pairs_rgb_depth=["visible", "depth"])


def native_meta(ds):
    """每个 native GT 的静态属性。"""
    meta = {}
    for i in range(len(ds)):
        lab = ds.labels[i]
        shape = lab.get("shape")
        if not shape:
            continue
        h0, w0 = float(shape[0]), float(shape[1])
        bb = np.asarray(lab.get("bboxes", np.zeros((0, 4))), dtype=np.float64)
        cls = np.asarray(lab.get("cls", []), dtype=np.int64)
        f = 1280.0 / max(h0, w0)                      # load_image 的长边缩放
        for k in range(len(cls)):
            if k >= len(bb):
                break
            nw, nh = bb[k, 2] * w0, bb[k, 3] * h0
            meta[INSTR.encode(i, k)] = dict(index=i, k=k, cls=int(cls[k]),
                                            ori_h=int(h0), ori_w=int(w0),
                                            native_w=float(nw), native_h=float(nh),
                                            native_area=float(nw * nh),
                                            work_area=float(nw * nh * f * f),
                                            layer=("HR" if h0 >= 1000 else "LR"))
    return meta


def bucket_sqrt(s):
    if s < 8:
        return "<8"
    if s < 12:
        return "8-12"
    if s < 18:
        return "12-18"
    if s < 24:
        return "18-24"
    if s < 32:
        return "24-32"
    if s < 96:
        return "32-96"
    return ">96"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays-on", type=int, default=100, help="mosaic-ON 的 Monte-Carlo epoch 数")
    ap.add_argument("--replays-off", type=int, default=100, help="mosaic-OFF 的 Monte-Carlo epoch 数")
    ap.add_argument("--out", default="_sge_replay.json")
    args = ap.parse_args()

    print("=" * 78)
    print("D′ SMALL-GT EFFECTIVE EXPOSURE — Monte-Carlo REPLAY（只读；不训练/不改任何东西）")
    print("=" * 78)

    INSTR.install()
    BaseDataset.load_image = _STUB_LOAD

    summary = {}
    all_agg = {}
    for mosaic_p, nrep, tag in ((1.0, args.replays_on, "mosaic_ON"),
                                (0.0, args.replays_off, "mosaic_OFF")):
        ds = build(mosaic_p)
        meta = native_meta(ds)
        n = len(ds)
        agg = dict(inst=defaultdict(int), inst_main=defaultdict(int), inst_tile=defaultdict(int),
                   mosaic=defaultdict(int), final=defaultdict(int), final_area=defaultdict(list),
                   ep_inst=defaultdict(int), ep_final=defaultdict(int),
                   FA_uid=[], FA_val=[], MA_uid=[], MA_val=[])
        st = State(agg)
        INSTR.set_on_stage(st)   # 逐 regime 切换回调；包装器只在 install() 时装一次

        print(f"\n--- {tag} (mosaic p={mosaic_p}) : {n} 图 × {nrep} epoch ---")
        for rep in range(nrep):
            random.seed(BASE_SEED + 7919 * rep + (0 if mosaic_p else 1))
            np.random.seed(BASE_SEED + 7919 * rep + (0 if mosaic_p else 1))
            order = list(range(n))
            random.shuffle(order)                    # 模拟 dataloader 的 epoch shuffle
            st.ep_inst = set(); st.ep_final = set()
            for i in order:
                st.n_native = 0
                ds[i]
            for u in st.ep_inst:
                agg["ep_inst"][u] += 1
            for u in st.ep_final:
                agg["ep_final"][u] += 1
            if (rep + 1) % max(1, nrep // 10) == 0:
                print(f"    ep {rep + 1}/{nrep}")
        all_agg[tag] = agg
        all_meta = meta
        summary[tag] = dict(n_images=n, n_replays=nrep, n_native_gt=len(meta),
                            mosaic_p=mosaic_p)
        print(f"    native GT 总数 = {len(meta)}；最终 target 出现次数 = {sum(agg['final'].values())}")

    # ---- 落盘（纯数组 npz，避免 JSON 体积）----
    payload = {}
    for tag, agg in all_agg.items():
        for k in ("inst", "inst_main", "inst_tile", "mosaic", "final", "ep_inst", "ep_final"):
            u = np.array(sorted(agg[k]), dtype=np.int64)
            v = np.array([agg[k][x] for x in u], dtype=np.int64)
            payload[f"{tag}__{k}__uid"] = u
            payload[f"{tag}__{k}__val"] = v
        for key, uk, vk in (("final_area", "FA_uid", "FA_val"), ("mosaic_area", "MA_uid", "MA_val")):
            payload[f"{tag}__{key}__uid"] = np.asarray(agg[uk], dtype=np.int64)
            payload[f"{tag}__{key}__flat"] = np.asarray(agg[vk], dtype=np.float32)
    np.savez_compressed(OUT / "_sge_replay.npz", **payload)

    (OUT / "_sge_native_meta.json").write_text(
        json.dumps({str(k): v for k, v in all_meta.items()}, ensure_ascii=False), encoding="utf-8")
    (OUT / "_sge_replay_meta.json").write_text(json.dumps(dict(
        base_seed=BASE_SEED, hyp=D_HYP, summary=summary,
        note="Monte-Carlo distribution estimate; training RNG stream is not reproducible."),
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {OUT / '_sge_replay.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
