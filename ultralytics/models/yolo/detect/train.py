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
