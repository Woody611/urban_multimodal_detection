"""_sge_verify.py — 插桩与打桩的正确性验证（必须 PASS 才允许跑全量 replay）

V1  STUB-EQUIVALENCE：用「同形状零图」替代真实像素（不读盘）跑同一条 pipeline，
    在**同 seed** 下 final cls / bboxes / img shape 必须逐位相同。
    依据：D′ 的像素级增强全部关闭（alb p=0、RandomHSV 对非 3ch 直接 return、
    degrees/shear/perspective=0，且 fliplr/flipud 是纯几何），几何只依赖 img.shape。
    该依据必须**实测**，不能靠推理。

V2  UID-CORRECTNESS：final 每个 GT 的 uid 解出 (source_index, k)，其 native 类别
    必须等于 final 类别。类别来自完全独立的代码路径 ⇒ 独立正确性检验。

V3  NO-DESYNC：uid 长度必须处处等于 instances 长度。
"""
from __future__ import annotations

import json
import math
import random
import sys
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

# D′ 真实 runtime hyp（runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/args.yaml）
D_HYP = dict(imgsz=1280, task="detect", rect=False, cache=False, single_cls=False,
             classes=None, fraction=1.0, channels=5, use_simotm="RGBID",
             object_scale_aug=False, mixup=0.0, copy_paste=0.0,
             degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
             flipud=0.0, fliplr=0.5)

# ---- 打桩开关（dispatcher 避免污染已建好的 real dataset）----
_ORIG_LOAD = BaseDataset.load_image
_STUB_LOAD = {"fn": INSTR.make_stub_load_image(5)}
_USE_STUB = {"on": False}


def _dispatcher(self, i, rect_mode=True):
    return _STUB_LOAD["fn"](self, i, rect_mode) if _USE_STUB["on"] else _ORIG_LOAD(self, i, rect_mode)


class _Ind:
    """可切换的 on_stage 回调（install 时传入，逐样本改 .fn）。"""

    def __init__(self):
        self.fn = None

    def __call__(self, name, labels, uid):
        if self.fn is not None:
            self.fn(name, labels, uid)


_IND = _Ind()


def build(mosaic_p):
    data = yaml_load(str(DS_YAML))
    cfg = get_cfg(overrides=dict(D_HYP, mosaic=mosaic_p))
    img_path = (Path(data["path"]) / data["train"])
    return build_yolo_dataset(cfg, str(img_path), 8, data, mode="train", use_simotm="RGBID",
                              pairs_rgb_ir=["visible", "infrared"], pairs_rgb_depth=["visible", "depth"])


def capture(ds, i, seed):
    """跑一次，返回 {stage: dict(uid,bboxes,cls,img_shape)} 与 final label。"""
    snap = {}

    def hook(name, labels, uid):
        inst = labels.get("instances")
        if inst is None:
            return
        snap[name] = dict(uid=None if uid is None else np.asarray(uid).copy(),
                          bboxes=np.asarray(inst.bboxes).copy(),
                          cls=np.asarray(labels.get("cls", []), dtype=np.int64).copy(),
                          img_shape=tuple(np.shape(labels.get("img", np.zeros((0, 0))))[:2]))

    _IND.fn = hook
    try:
        random.seed(seed)
        np.random.seed(seed)
        lab = ds[i]
    finally:
        _IND.fn = None
    return snap, lab


def main():
    print("=" * 78)
    print("SGE VERIFY — 插桩 / 打桩正确性（PASS 才允许跑全量 replay）")
    print("=" * 78)
    INSTR.install(on_stage=_IND)
    BaseDataset.load_image = _dispatcher

    v1, v2 = [], []
    for stub_ch, mosaic_p, regime in ((5, 1.0, "mosaic-ON/5ch"), (5, 0.0, "mosaic-OFF/5ch"),
                                      (1, 1.0, "mosaic-ON/1ch"), (1, 0.0, "mosaic-OFF/1ch")):
        _STUB_LOAD["fn"] = INSTR.make_stub_load_image(stub_ch)
        print(f"\n--- {regime} ---")
        ds = build(mosaic_p)
        n = len(ds)
        shapes = [(ds.labels[i].get("shape") or (0, 0))[0] for i in range(n)]
        hr = [i for i in range(n) if shapes[i] >= 1000]
        lr = [i for i in range(n) if 0 < shapes[i] < 1000]
        idxs = sorted({*[hr[k] for k in range(0, len(hr), max(1, len(hr) // 12))][:12],
                       *[lr[k] for k in range(0, len(lr), max(1, len(lr) // 12))][:12]})
        print(f"  样本 {len(idxs)}（HR {sum(1 for i in idxs if shapes[i] >= 1000)} / "
              f"LR {sum(1 for i in idxs if shapes[i] < 1000)}）；数据集 n={n}")

        for i in idxs:
            seed = 20260928 + i
            _USE_STUB["on"] = False
            snap_r, lab_r = capture(ds, i, seed)
            _USE_STUB["on"] = True
            snap_s, lab_s = capture(ds, i, seed)
            _USE_STUB["on"] = False

            v1.append(dict(i=int(i), regime=regime,
                           cls=bool(np.array_equal(lab_r["cls"], lab_s["cls"])),
                           bbox=bool(np.asarray(lab_r["bboxes"]).shape == np.asarray(lab_s["bboxes"]).shape
                                     and np.array_equal(np.asarray(lab_r["bboxes"]), np.asarray(lab_s["bboxes"]))),
                           img=bool(tuple(np.shape(lab_r["img"])[:2]) == tuple(np.shape(lab_s["img"])[:2])),
                           ch_r=int(np.shape(lab_r["img"])[-1]), ch_s=int(np.shape(lab_s["img"])[-1]),
                           n_gt=int(len(lab_r["cls"])),
                           n_mosaic=int(len(snap_r.get("mosaic", {}).get("cls", [])))))

            f = snap_r.get("final")
            if f is not None and f["uid"] is not None and len(f["uid"]) == len(f["cls"]):
                src, k = INSTR.decode(f["uid"])
                ok = sum(1 for j in range(len(f["cls"]))
                         if (lambda sc: sc is not None and int(k[j]) < len(sc)
                             and int(sc[int(k[j])]) == int(f["cls"][j]))(ds.labels[int(src[j])].get("cls")))
                v2.append(dict(i=int(i), regime=regime, n=int(len(f["cls"])), ok=int(ok)))
            else:
                v2.append(dict(i=int(i), regime=regime, n=-1, ok=-1))

    v3 = dict(desync=INSTR.STATS["desync"], where=INSTR.STATS["desync_where"],
              concat_no_uid=INSTR.STATS["concat_no_uid"], rp_pending_miss=INSTR.STATS["rp_pending_miss"])

    n_cls = sum(r["cls"] for r in v1); n_bb = sum(r["bbox"] for r in v1); n_im = sum(r["img"] for r in v1)
    vv = [r for r in v2 if r["n"] >= 0]
    tot_gt = sum(r["n"] for r in vv); tot_ok = sum(r["ok"] for r in vv)

    print("\n--- V1 STUB-EQUIVALENCE ---")
    print(f"  cls 逐位相同  : {n_cls}/{len(v1)}")
    print(f"  bbox 逐位相同 : {n_bb}/{len(v1)}")
    print(f"  img shape 相同: {n_im}/{len(v1)}")
    print("--- V2 UID-CORRECTNESS ---")
    print(f"  类别一致: {tot_ok}/{tot_gt} ({100.0 * tot_ok / max(tot_gt, 1):.3f}%) over {len(vv)} samples")
    print("--- V3 DESYNC ---")
    print(f"  {v3}")

    ok = (n_cls == n_bb == n_im == len(v1)) and tot_ok == tot_gt and tot_gt > 0 and INSTR.STATS["desync"] == 0
    print("\n" + "=" * 78)
    print("VERIFY PASS" if ok else "VERIFY FAIL")
    print("=" * 78)
    (OUT / "_sge_verify.json").write_text(json.dumps(dict(v1=v1, v2=v2, v3=v3, pass_=ok),
                                                     indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
