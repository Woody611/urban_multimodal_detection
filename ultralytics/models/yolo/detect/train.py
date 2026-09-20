# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


def _remap_separate_stem(model, src, rgb_stem, aux_stems):
    """Explicit pretrained remap for the Separate-Stem fusion layout (3ch + 1ch + 1ch).

    Layout (see ``configs/yolo11m_sepstem.yaml``)::

        Silence -> SilenceChannel[0,3] -> Conv(3->C_rgb) -\
                -> SilenceChannel[3,4] -> Conv(1->C_ir)   >- Concat -> stock YOLO11 backbone
                -> SilenceChannel[4,5] -> Conv(1->C_d)  -/

    Deterministic rules — no index guessing, no silent fallback:

    * **RGB stem** (in=3, out=C_rgb) <- stock stem ``model.0`` **output-channel prefix**
      ``[:C_rgb]`` (conv weight and all BN params).
    * **IR / Depth stems** (in=1, out=C_*) <- grayscale-reduced stock stem
      ``mean(stock_w[:, 0:3], dim=1, keepdim=True)`` then output-channel prefix ``[:C_*]``.
      Which aux stem is IR vs Depth is read from the ``c_start`` of the ``SilenceChannel``
      that *feeds* it (3 = infrared, 4 = depth) — never from module position.
    * Every layer after the fusion ``Concat`` is copied from stock layer
      ``(i - concat_idx)``; ``concat_idx`` is **derived from the yaml**, not hardcoded.

    Raises ``RuntimeError`` on any missing / mismatched / ambiguous item. This function
    never leaves a tensor silently randomly initialised.
    """
    layers = list(model.yaml.get("backbone", [])) + list(model.yaml.get("head", []))

    # ---- locate the fusion Concat (the three-input one) ----
    cat_idx = None
    for i, lyr in enumerate(layers):
        if str(lyr[2]) == "Concat" and isinstance(lyr[0], (list, tuple)) and len(lyr[0]) == 3:
            cat_idx = i
            break
    if cat_idx is None:
        raise RuntimeError(
            "[sepstem remap] could not locate a 3-input fusion Concat in the model yaml; refusing to guess"
        )
    offset = cat_idx  # candidate layer i (i > offset)  <->  stock layer (i - offset)

    idx_of = {id(m): i for i, m in enumerate(model.model)}

    # ---- the Concat must exactly reference the three stem layers ----
    expected_from = sorted(idx_of[id(m)] for m in [rgb_stem] + list(aux_stems))
    got_from = sorted(layers[cat_idx][0])
    if got_from != expected_from:
        raise RuntimeError(
            f"[sepstem remap] fusion Concat at layer {cat_idx} references {got_from}, "
            f"but the detected stems are at {expected_from}; refusing to guess"
        )

    # ---- identify IR vs Depth from each aux stem's feeder SilenceChannel ----
    def _feeder_c_start(i):
        fr = layers[i][0]
        j = i - 1 if fr == -1 else fr
        if not isinstance(j, int) or j < 0 or j >= len(layers):
            return None
        mod = model.model[j]
        return int(getattr(mod, "c_start", -1)) if type(mod).__name__ == "SilenceChannel" else None

    ir_stem = depth_stem = None
    for m in aux_stems:
        i = idx_of[id(m)]
        cs = _feeder_c_start(i)
        if cs == 3:
            ir_stem = m
        elif cs == 4:
            depth_stem = m
        else:
            raise RuntimeError(
                f"[sepstem remap] aux stem at layer {i} is not fed by a SilenceChannel with "
                f"c_start in (3, 4) (got {cs}); refusing to guess IR vs Depth"
            )
    if ir_stem is None or depth_stem is None:
        raise RuntimeError("[sepstem remap] failed to identify both IR and Depth stems")

    # ---- stock stem ----
    stock_w = src.get("model.0.conv.weight")
    if stock_w is None:
        raise RuntimeError("[sepstem remap] stock stem 'model.0.conv.weight' not present in pretrained weights")
    if stock_w.shape[1] != 3:
        raise RuntimeError(f"[sepstem remap] expected a 3-channel stock stem, got in_channels={stock_w.shape[1]}")
    stock_out = stock_w.shape[0]

    n = 0
    with torch.no_grad():
        # ---------- the three stems ----------
        for stem, kind in ((rgb_stem, "rgb"), (ir_stem, "ir"), (depth_stem, "depth")):
            li = idx_of[id(stem)]
            out_c = stem.conv.weight.shape[0]
            if out_c > stock_out:
                raise RuntimeError(
                    f"[sepstem remap] {kind} stem out={out_c} exceeds stock stem out={stock_out}"
                )
            if kind == "rgb":
                if stem.conv.in_channels != 3:
                    raise RuntimeError(f"[sepstem remap] RGB stem in_channels={stem.conv.in_channels}, expected 3")
                new_w = stock_w[:out_c].clone()
            else:
                if stem.conv.in_channels != 1:
                    raise RuntimeError(f"[sepstem remap] {kind} stem in_channels={stem.conv.in_channels}, expected 1")
                # Deterministic grayscale reduction, then output-channel prefix.
                # Computed in float32 regardless of the source checkpoint dtype (yolo11m.pt
                # stores fp16), so the result is bit-reproducible and testable.
                new_w = (
                    stock_w[:out_c].float().mean(dim=1, keepdim=True).to(stem.conv.weight.dtype).clone()
                )
            stem.conv.weight.copy_(new_w)
            n += 1
            for bk, bv in stem.bn.state_dict().items():
                sk = f"model.0.bn.{bk}"
                if sk not in src:
                    raise RuntimeError(f"[sepstem remap] stock '{sk}' missing for {kind} stem BN")
                sv = src[sk]
                if sv.ndim == 0:
                    bv.copy_(sv)
                else:
                    if sv.shape[0] < out_c:
                        raise RuntimeError(f"[sepstem remap] stock '{sk}' has {sv.shape[0]} < {out_c} entries")
                    bv.copy_(sv[:out_c])
                n += 1
            LOGGER.info(f"sepstem remap: {kind} stem (layer {li}, out={out_c}) initialised")

        # ---------- everything after the fusion Concat ----------
        csd = model.state_dict()
        detect_idx = len(layers) - 1
        nc = int(getattr(model.model[-1], "nc", -1))
        n_cls_untrained = 0
        with torch.no_grad():
            for i in range(offset + 1, len(layers)):
                s = i - offset
                prefix = f"model.{i}."
                for k, v in csd.items():
                    if not k.startswith(prefix) or v.ndim == 0:
                        continue
                    sk = f"model.{s}." + k[len(prefix) :]
                    sv = src.get(sk)
                    if sv is not None and tuple(sv.shape) == tuple(v.shape):
                        v.copy_(sv)
                        n += 1
                        continue
                    # The ONLY legitimately non-transferable block: Detect's class branch
                    # (cv3). The source checkpoint was trained with a different number of
                    # classes, so its cls head is re-initialised for this dataset — the same
                    # thing stock ultralytics does for every new-nc fine-tune.
                    # Declared explicitly, constrained by shape/nc assertions, and logged;
                    # anything else raises.
                    if (
                        i == detect_idx
                        and ".cv3." in k
                        and sv is not None
                        and tuple(sv.shape[1:]) == tuple(v.shape[1:])
                        and v.shape[0] == nc
                        and sv.shape[0] != nc
                    ):
                        n_cls_untrained += 1
                        continue
                    if sv is None:
                        raise RuntimeError(f"[sepstem remap] no pretrained counterpart for '{k}' (looked for '{sk}')")
                    raise RuntimeError(
                        f"[sepstem remap] shape mismatch for '{k}': candidate {tuple(v.shape)} vs stock {sk} {tuple(sv.shape)}"
                    )

    LOGGER.info(
        f"Separate-stem pretrained remap: {n} tensors transferred "
        f"(3 stems + layers {offset + 1}..{len(layers) - 1} <- stock 1..{len(layers) - 1 - offset}); "
        f"Detect cv3 class branch re-initialised for nc={nc}: {n_cls_untrained} tensors not transferred"
    )
    return n


def _transfer_rgb_pretrained(model, weights):
    """Transfer single-branch COCO pretrained weights into fusion models.

    The default ``model.load()`` matches weights by (key-name, shape) via
    ``intersect_dicts``, so any stem whose input-channel count differs from the
    pretrained 3ch is silently skipped. Two fusion families are handled here:

    - RGBD mid-fusion (separate 1ch depth stem): the RGB branch is prefixed by
      ``Silence``/``SilenceChannel``, shifting its indices by +2, so ``load()``
      matches ~0 RGB keys. Remap module-by-module by structural correspondence,
      then init the depth stem with ``W_depth = mean(W_R, W_G, W_B)``.
    - RGBID early fusion (single 5ch stem): every layer except the first Conv
      matches ``load()`` exactly; only the first Conv weight (5ch vs 3ch) is
      skipped. Copy the pretrained RGB weights into the first 3 channels and init
      the IR/depth channels with ``mean(W_R, W_G, W_B)``.

    Single-branch models (no 1ch or 5ch stem) are left untouched: standard
    ``load()`` already handles them. Returns the number of tensors transferred.
    """
    from pathlib import Path

    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.tasks import attempt_load_one_weight

    stems = [m for m in model.model if isinstance(m, Conv)]

    # source single-branch state_dict
    if isinstance(weights, (str, Path)):
        src_model, _ = attempt_load_one_weight(weights)
        src = src_model.state_dict()
    elif isinstance(weights, dict):
        src = weights["model"] if "model" in weights else weights
    else:
        src = weights.state_dict()

    # ---- RGBID early fusion: first Conv absorbs all 5 channels (no 1ch stem) ----
    five_ch = next((m for m in stems if getattr(m.conv, "in_channels", None) == 5), None)
    if five_ch is not None and not any(getattr(m.conv, "in_channels", None) == 1 for m in stems):
        # Early fusion has no Silence -> first Conv is model.0. Standard load() already
        # matched every key except the 5ch stem weight; fill it here (RGB=pretrained,
        # IR/D=mean(R,G,B), mirroring the RGBD depth-stem init convention).
        w = src.get("model.0.conv.weight")
        if w is not None and w.shape[1] == 3 and w.shape[0] == five_ch.conv.weight.shape[0]:
            n_aux = five_ch.conv.weight.shape[1] - 3
            with torch.no_grad():
                five_ch.conv.weight[:, :3].copy_(w)  # RGB 预训练
                five_ch.conv.weight[:, 3:].copy_(
                    w.mean(dim=1, keepdim=True).repeat(1, n_aux, 1, 1)
                )  # IR/D = mean(R,G,B)
            LOGGER.info(
                f"RGBID early-fusion stem init: RGB=pretrained, IR/D=mean(R,G,B) "
                f"(stem in={five_ch.conv.in_channels})"
            )
            return 5

    # ---- Separate-Stem fusion (exactly one 3ch stem + exactly two 1ch stems) ----
    # Must be dispatched BEFORE the RGBD branch below: that branch keys off the mere
    # presence of a 1ch stem and would otherwise swallow this layout, remapping through a
    # hardcoded concat_res index table that does not correspond to it.
    _three = [m for m in stems if getattr(m.conv, "in_channels", None) == 3]
    _ones = [m for m in stems if getattr(m.conv, "in_channels", None) == 1]
    if len(_three) == 1 and len(_ones) == 2:
        return _remap_separate_stem(model, src, _three[0], _ones)

    if not any(getattr(m.conv, "in_channels", None) == 1 for m in stems):
        return 0  # not an RGBD fusion model -> nothing to remap

    # (fusion module idx -> single-branch yolo11 idx) for concat_res mid-fusion.
    # The layout shifts between the 3-scale (Detect P3/P4/P5) and 4-scale (P2
    # variant) architectures, so select by the number of detection heads.
    nl = getattr(model.model[-1], "nl", 3)
    if nl == 4:  # Detect(P2, P3, P4, P5)
        pairs = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 19: 5, 20: 6, 26: 7, 27: 8,
                 33: 9, 34: 10, 37: 13, 40: 16, 44: 17, 46: 19, 47: 20,
                 49: 22, 50: 23}
    else:  # Detect(P3, P4, P5)
        pairs = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 16: 5, 17: 6, 23: 7, 24: 8,
                 30: 9, 31: 10, 34: 13, 37: 16, 38: 17, 40: 19, 41: 20,
                 43: 22, 44: 23}
    n = 0
    with torch.no_grad():
        for fi, si in pairs.items():
            prefix = f"model.{fi}."
            for fname, fparam in model.state_dict().items():
                if fname.startswith(prefix) and fparam.ndim > 0:
                    sname = f"model.{si}." + fname[len(prefix):]
                    sv = src.get(sname)
                    if sv is not None and sv.shape == fparam.shape:
                        fparam.copy_(sv)
                        n += 1

    # depth stem: W_depth = mean(W_R, W_G, W_B)
    rgb = next((m for m in stems if m.conv.in_channels == 3), None)
    depth = next((m for m in stems if m.conv.in_channels == 1), None)
    if rgb is not None and depth is not None and rgb.conv.weight.shape[0] == depth.conv.weight.shape[0]:
        with torch.no_grad():
            depth.conv.weight.copy_(rgb.conv.weight.mean(dim=1, keepdim=True))
            if depth.conv.bias is not None and rgb.conv.bias is not None:
                depth.conv.bias.copy_(rgb.conv.bias)
        LOGGER.info(
            f"RGBD depth-stem init: W_depth = mean(W_R,W_G,W_B) "
            f"(rgb in={rgb.conv.in_channels} -> depth in={depth.conv.in_channels})"
        )
    LOGGER.info(f"RGBD pretrained remap: transferred {n} tensors")
    return n


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs, use_simotm=self.args.use_simotm,pairs_rgb_ir=self.args.pairs_rgb_ir,pairs_rgb_depth=self.args.pairs_rgb_depth)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights is None and isinstance(self.args.pretrained, (str, Path)):
            # `YOLO(yaml).train(pretrained=...)` reaches here with weights=None (self.ckpt
            # is empty for yaml-built models), which silently skips weight loading and the
            # RGBD remap below. Fall back to the trainer's `pretrained` arg so weights load.
            weights, _ = attempt_load_one_weight(self.args.pretrained)
        if weights:
            model.load(weights)
            _transfer_rgb_pretrained(model, weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
            use_simotm=self.args.use_simotm,  # 2025-01-05
        )
        # 'yzc' 2025-03-03
        if self.args.use_simotm in ("RGBT", "RGBRGB6C"):
            plot_images(
                images=batch["img"],
                batch_idx=batch["batch_idx"],
                cls=batch["cls"].squeeze(-1),
                bboxes=batch["bboxes"],
                paths=batch["im_file"],
                fname=self.save_dir / f"train_batch{ni}_ir.jpg",
                on_plot=self.on_plot,
                use_simotm=self.args.use_simotm,  # 2025-01-05
                ir_show=True  # 显示红外图像以及标签
            )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        """Get batch size by calculating memory occupation of model."""
        train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
        # 4 for mosaic augmentation
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        return super().auto_batch(max_num_obj)
