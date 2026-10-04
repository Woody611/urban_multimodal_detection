"""P3 — pre-NMS / NMS / top-100 disappearance audit: instrumented inference dump.

READ-ONLY w.r.t. model & code: the model is never modified, no training, no parameter
sweep.  This script re-runs the *production* rect inference path
(`scripts/predict_rect.py`) verbatim, and additionally records, per image:

  Layer A  pre-NMS candidates   : (xyxy_canvas, conf, cls, full 12-class score vector)
  Layer B  NMS kept / suppressed: provenance (suppressor id, IoU, same-class, confs, areas)
  Layer C  top-100              : derived from Layer B with the frozen evaluator's cap

Reproducibility gate: Layer B, scaled to native and formatted with the production
`_format_lines(..., max_boxes=300)`, must reproduce
`diagnostic/sepstem_clahe/best_full/results/*.txt` byte-for-byte.

  python diagnostic/p3_disappearance_audit/_run_infer.py --limit 8 --check
  python diagnostic/p3_disappearance_audit/_run_infer.py            # full 400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from predict import _to_chw, _format_lines, _load_yaml  # noqa: E402
from predict_rect import _preprocess_rect  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.loaders import LoadImagesAndVideos  # noqa: E402
from ultralytics.utils.ops import non_max_suppression, scale_boxes, xywh2xyxy  # noqa: E402
from torchvision.ops import nms as tv_nms  # noqa: E402

OUT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
TRAIN_CFG = ROOT / "configs/train_rgbid_sepstem_clahe.yaml"
SOURCE = ROOT / "data/processed/rgbid_split/images/val/visible"
REFERENCE = ROOT / "diagnostic/sepstem_clahe/best_full/results"

CONF, IOU, IMGSZ, BATCH, MAX_DET = 0.001, 0.7, 1280, 16, 300
MAX_NMS = 30000
MAX_WH = 7680  # ultralytics class offset for class-aware NMS

PROVENANCE_FILES = [
    "scripts/predict_rect.py", "scripts/predict.py",
    "ultralytics/utils/ops.py", "ultralytics/data/loaders.py",
    "scripts/official_eval.py",
    "configs/train_rgbid_sepstem_clahe.yaml", "configs/yolo11m_sepstem.yaml",
]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pre_nms_matrix(preds_k: torch.Tensor, nc: int):
    """Replicate ultralytics non_max_suppression's candidate construction for one image.

    Returns (A[n,6] = xyxy,conf,cls on canvas, C[n,nc] = full class score vector, n_capped: bool)
    """
    p = preds_k.transpose(-1, -2)                       # (A, 4+nc)
    p = torch.cat((xywh2xyxy(p[..., :4]), p[..., 4:]), dim=-1)
    xc = p[:, 4:4 + nc].amax(1) > CONF
    p = p[xc]
    if not p.shape[0]:
        return p.new_zeros((0, 6)), p.new_zeros((0, nc)), False
    box, cls = p.split((4, nc), 1)
    i, j = torch.where(cls > CONF)                      # multi_label expansion
    A = torch.cat((box[i], p[i, 4 + j, None], j[:, None].float()), 1)
    all_cls = cls[i]                                    # full class scores of the candidate box
    capped = False
    if A.shape[0] > MAX_NMS:
        keep = A[:, 4].argsort(descending=True)[:MAX_NMS]
        A, all_cls, capped = A[keep], all_cls[keep], True
    return A, all_cls, capped


def nms_provenance(A: torch.Tensor, iou_thres: float, max_wh: int = MAX_WH):
    """Run the real torchvision NMS; then compute the suppressor of every dropped candidate.

    Returns dict with kept_full (indices kept by NMS, score order), kept (after max_det),
    and suppression records for the rest.
    """
    if A.shape[0] == 0:
        return dict(kept_full=np.zeros(0, int), kept=np.zeros(0, int), recs=[])
    cls = A[:, 5:6].long()
    boxes = A[:, :4] + cls.float() * max_wh
    scores = A[:, 4]
    order = scores.argsort(descending=True)
    i_full = tv_nms(boxes, scores, iou_thres).numpy()   # score-descending
    kept = i_full[:MAX_DET]
    kept_set = set(i_full.tolist())

    # greedy suppression provenance, reproducing torchvision's keep rule
    recs = []
    if len(i_full):
        kept_sort = i_full                                   # already score-desc
        for idx in order.numpy().tolist():
            if idx in kept_set:
                continue
            a = A[idx]
            best = None
            for k in kept_sort:
                if A[k, 5].item() != a[5].item():            # class-aware: cross-class never suppresses
                    continue
                if A[k, 4].item() < a[4].item():             # only higher-scoring boxes can suppress
                    continue
                b = A[k]
                lt = np.maximum(a[:2].numpy(), b[:2].numpy())
                rb = np.minimum(a[2:4].numpy(), b[2:4].numpy())
                wh = np.clip(rb - lt, 0, None)
                inter = float(wh[0] * wh[1])
                aa = float((a[2] - a[0]) * (a[3] - a[1]))
                ba = float((b[2] - b[0]) * (b[3] - b[1]))
                iou = inter / (aa + ba - inter + 1e-12)
                if iou > iou_thres and (best is None or b[4].item() > best[1]):
                    best = (int(k), float(iou))
            recs.append(dict(idx=int(idx), suppressor=-1 if best is None else best[0],
                             iou=0.0 if best is None else best[1],
                             capped_by_maxdet=False))
    # candidates kept by NMS but cut by i[:max_det]
    for idx in i_full[MAX_DET:].tolist():
        recs.append(dict(idx=int(idx), suppressor=-2, iou=0.0, capped_by_maxdet=True))
    return dict(kept_full=i_full, kept=np.asarray(kept), recs=recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--check", action="store_true", help="verify Layer B reproduces reference TXT")
    ap.add_argument("--out", default="layers.npz")
    a = ap.parse_args()

    tcfg = _load_yaml(TRAIN_CFG)
    use_simotm = str(tcfg.get("use_simotm", "RGBID"))
    ir_encoding = str(tcfg.get("ir_encoding", "percentile") or "percentile")
    pairs_ir = list(tcfg.get("pairs_rgb_ir", ["visible", "infrared"]))
    pairs_d = list(tcfg.get("pairs_rgb_depth", ["visible", "depth"]))

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = YOLO(str(WEIGHTS))
    nn = model.model.eval()
    nc = int(getattr(nn, "nc", 12))
    stride = int(nn.stride.max())

    loader = LoadImagesAndVideos(str(SOURCE), batch=BATCH, use_simotm=use_simotm, imgsz=IMGSZ,
                                 pairs_rgb_ir=pairs_ir, pairs_rgb_depth=pairs_d,
                                 ir_encoding=ir_encoding)

    rec = {k: [] for k in ["stem", "orig_h", "orig_w", "canvas_h", "canvas_w",
                           "nA", "nB", "ref_lines"]}
    layerA, layerB, supp = [], [], []
    a_off, b_off, s_off = [], [], []
    t0 = time.time()
    done = 0
    for paths, imgs, _info in loader:
        pre = [_preprocess_rect(im, IMGSZ, stride) for im in imgs]
        groups: dict = {}
        for i, (padded, *_r) in enumerate(pre):
            groups.setdefault(padded.shape[:2], []).append(i)
        outp = [None] * len(imgs)
        layers = [None] * len(imgs)
        for shape, idxs in groups.items():
            x = torch.stack([torch.from_numpy(_to_chw(pre[j][0])) for j in idxs]).float() / 255.0
            with torch.no_grad():
                preds = nn(x)
            # Detect.forward returns (y, x) in eval mode; non_max_suppression unpacks [0]
            y = preds[0] if isinstance(preds, (list, tuple)) else preds
            out = non_max_suppression(y, CONF, IOU, nc=nc, max_det=MAX_DET,
                                      multi_label=True, in_place=False)
            for k, j in enumerate(idxs):
                o = out[k]
                if o is not None and len(o):
                    o = o.clone()
                    o[:, :4] = scale_boxes(x.shape[2:], o[:, :4], pre[j][2], ratio_pad=pre[j][1])
                outp[j] = o
                # ---- layer A + provenance on canvas coords, then scale ----
                A, allc, capped = pre_nms_matrix(y[k], nc)
                prov = nms_provenance(A, IOU)
                if A.shape[0]:
                    A_nat = A.clone()
                    A_nat[:, :4] = scale_boxes(x.shape[2:], A_nat[:, :4], pre[j][2],
                                               ratio_pad=pre[j][1])
                else:
                    A_nat = A
                layers[j] = dict(A=A_nat.numpy(), all_cls=allc.numpy(),
                                 kept=prov["kept"], kept_full=prov["kept_full"],
                                 recs=prov["recs"], capped=capped)
        for i, p in enumerate(paths):
            L = layers[i]
            rec["stem"].append(Path(p).stem)
            rec["orig_h"].append(pre[i][2][0]); rec["orig_w"].append(pre[i][2][1])
            rec["canvas_h"].append(pre[i][0].shape[0]); rec["canvas_w"].append(pre[i][0].shape[1])
            rec["nA"].append(int(L["A"].shape[0]))
            rec["nB"].append(int(0 if outp[i] is None else len(outp[i])))
            a_off.append(sum(x.shape[0] for x in layerA)); layerA.append(L["A"])
            b_off.append(sum(x.shape[0] for x in layerB))
            layerB.append(outp[i].numpy() if outp[i] is not None else np.zeros((0, 6), np.float32))
            # global-indexed suppression records
            for r in L["recs"]:
                supp.append(dict(stem=Path(p).stem, idx=r["idx"],
                                 suppressor_a=(-1 if r["suppressor"] in (-1, -2) else a_off[-1] + r["suppressor"]),
                                 iou=r["iou"], capped_by_maxdet=r["capped_by_maxdet"]))
            if a.check:
                lines = _format_lines(outp[i], pre[i][2][0], pre[i][2][1], MAX_DET)
                ref = REFERENCE / f"{Path(p).stem}.txt"
                rec["ref_lines"].append(
                    (("\n".join(lines) + ("\n" if lines else "")) ==
                     (ref.read_text(encoding="utf-8") if ref.exists() else "")))
        done += len(paths)
        el = time.time() - t0
        print(f"[infer] {done} imgs  {el:.1f}s  {done/max(el,1e-9):.2f} img/s", flush=True)
        if a.limit and done >= a.limit:
            break

    savez = dict(
        stem=np.array(rec["stem"]),
        orig_h=np.array(rec["orig_h"], np.int32), orig_w=np.array(rec["orig_w"], np.int32),
        canvas_h=np.array(rec["canvas_h"], np.int32), canvas_w=np.array(rec["canvas_w"], np.int32),
        nA=np.array(rec["nA"], np.int32), nB=np.array(rec["nB"], np.int32),
        layerA=np.concatenate(layerA, 0) if layerA else np.zeros((0, 6), np.float32),
        layerB=np.concatenate(layerB, 0) if layerB else np.zeros((0, 6), np.float32),
        a_off=np.array(a_off, np.int64), b_off=np.array(b_off, np.int64),
        supp_idx=np.array([s["idx"] for s in supp], np.int64),
        supp_a=np.array([s["suppressor_a"] for s in supp], np.int64),
        supp_iou=np.array([s["iou"] for s in supp], np.float64),
        supp_maxdet=np.array([s["capped_by_maxdet"] for s in supp], bool),
        supp_stem=np.array([s["stem"] for s in supp]),
    )
    np.savez_compressed(OUT / a.out, **savez)
    prov = {f: sha256(ROOT / f) for f in PROVENANCE_FILES}
    prov["checkpoint_best.pt"] = sha256(WEIGHTS)
    meta = dict(git_sha=None, inference=dict(conf=CONF, iou=IOU, imgsz=IMGSZ, batch=BATCH,
                                             max_det=MAX_DET, mode="rect", scaleup=True,
                                             multi_label=True, agnostic=False,
                                             max_nms=MAX_NMS, max_wh=MAX_WH),
                source=str(SOURCE.relative_to(ROOT)), nc=nc, stride=stride,
                use_simotm=use_simotm, ir_encoding=ir_encoding,
                provenance_sha256=prov, images=len(rec["stem"]),
                reference=str(REFERENCE.relative_to(ROOT)),
                reference_reproduced=(all(rec["ref_lines"]) if a.check else None),
                n_ref_checked=len(rec["ref_lines"]))
    (OUT / f"meta_{Path(a.out).stem}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("[infer] reference reproduced:", meta["reference_reproduced"],
          f"({meta['n_ref_checked']} files)")
    print("[infer] saved", OUT / a.out)


if __name__ == "__main__":
    main()
