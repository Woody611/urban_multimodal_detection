"""_sge_instrument.py — native-GT lineage 插桩（SMALL_GT_EXPOSURE_AUDIT）

目的：在**真实** D′ train augmentation pipeline 上，把每个 native GT 从
`get_image_and_label` 一路追到 `Format` 输出（= 真正进 loss 的 GT），拿到
native → Mosaic → RandomPerspective → final 的逐阶段存活与几何。

为什么必须插桩（而不是像 V5 那样按输出 GT 分层）：
    V5 (`diagnostic/small_object_train_replay/_v5_replay.py`) 已实测「输出 GT 数 == Σ tile GT 数」
    在 244/250 个样本上不成立，因此它放弃了身份映射、改为对**增强后的输出 GT**分层。
    于是它回答的是「final 已经小的 GT 监督好不好」，而不是
    「native 小的 GT 有多少活到了 final」。本次要回答后者，必须有身份。

原理（不重写 augmentation，只挂载）：
    · 在每个源图上，`get_image_and_label` 返回的 instances 被打上 uid = index*GT_STRIDE + k
    · `Instances.concatenate / __getitem__ / remove_zero_area_boxes` 被包装成**同步搬运 uid**
      —— 这三处正是 Mosaic 拼接、RandomPerspective 过滤、Mosaic 零面积过滤的发生地
    · `RandomPerspective.__call__` 内部 `new_instances = Instances(...)` 是**新建对象**，
      用 PENDING_UID 让新对象继承输入 uid（长度相同，构造性成立）
    · 任何一步 uid 与 instances 长度不一致 ⇒ 记为 DESYNC（不猜、不修补）

不改变任何计算：所有包装都是「先调原函数，再搬运 uid」。唯一例外是把
`BoxCandidates` 的过滤结果按原样同步到 uid——那也是原函数自己的返回值。
"""
from __future__ import annotations

import math

import numpy as np

from ultralytics.utils.instance import Instances

GT_STRIDE = 100000          # uid = index * GT_STRIDE + k
UNKNOWN = np.int64(-1)

# 统计（供审计报告使用）
STATS = {"desync": 0, "desync_where": {}, "concat_no_uid": 0, "rp_pending_miss": 0}

# 回调持有者：install() 之后仍可用 set_on_stage() 逐样本替换，无需重装包装器
_ON_STAGE = {"fn": None}


def set_on_stage(fn):
    _ON_STAGE["fn"] = fn


def encode(index: int, k: int) -> int:
    return int(index) * GT_STRIDE + int(k)


def decode(uid):
    uid = np.asarray(uid, dtype=np.int64)
    return uid // GT_STRIDE, uid % GT_STRIDE


def _note_desync(where):
    STATS["desync"] += 1
    STATS["desync_where"][where] = STATS["desync_where"].get(where, 0) + 1


# ---------------------------------------------------------------- 打桩 load_image
def make_stub_load_image(channels=5):
    """返回一个与 `BaseDataset.load_image` **行为等价但零磁盘 I/O** 的替代实现。

    逐行镜像 `ultralytics/data/base.py:435-472`：同样的长边缩放、同样的
    `self.ims/self.im_hw0/self.im_hw` 缓存写回、**同样的 `self.buffer` FIFO 维护**。
    buffer 必须维护：`Mosaic.get_indexes` 用 `random.choices(buffer, k=3)` 选 tile，
    不维护 buffer 会直接 IndexError（首版即因此崩掉），而且会丢掉
    「mosaic tile 来自**最近加载过**的 64 张图」这一真实机制。

    几何等价性已由 `_sge_verify.py` V1 实测（48/48 逐位相同）。

    `channels`：几何只依赖 `img.shape[:2]`（Mosaic 放置 / RandomPerspective 的 M /
    box_candidates 都只用形状与 bbox），通道数不影响任何几何决策；而 cv2.warpAffine 在
    2560² 画布上是内存带宽瓶颈，5ch→1ch 实测 **13×** 加速（122.7→9.5 ms/sample）
    且 bbox 逐位相同。1ch 仅用于几何/存活统计；需要真实像素的监督 pass 不用 stub。
    """

    def _stub(self, i, rect_mode=True):
        im = self.ims[i]
        if im is None:
            shape = self.labels[i].get("shape")
            if not shape:
                raise KeyError(f"labels[{i}] has no 'shape'; cannot stub load_image")
            h0, w0 = int(shape[0]), int(shape[1])
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)
                w, h = ((min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                        if r != 1 else (w0, h0))
            else:
                w, h = (self.imgsz, self.imgsz) if not (h0 == w0 == self.imgsz) else (w0, h0)
            im = np.zeros((h, w, int(channels)), np.uint8)
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
            return im, (h0, w0), im.shape[:2]
        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    return _stub


def _check(self, where):
    """uid 长度必须与 instances 长度一致；否则记 DESYNC 并把 uid 置为 UNKNOWN。"""
    u = getattr(self, "uid", None)
    if u is None:
        return
    if len(u) != len(self):
        _note_desync(where)
        self.uid = None


# ---------------------------------------------------------------- 安装插桩
def install(on_stage=None, seed_getter=None):
    """安装插桩。

    on_stage(name, labels, uid) 可选回调，在每个阶段结束时被调用，用于抓取阶段快照。
    必须在构造 dataset / 跑 pipeline **之前**调用一次。
    """
    if on_stage is not None:
        _ON_STAGE["fn"] = on_stage
    PENDING = {"uid": None}

    # ---- 1) Instances.__init__：让 RandomPerspective 内部新建的 Instances 继承 uid ----
    _orig_init = Instances.__init__

    def _init(self, bboxes, segments=None, keypoints=None, bbox_format="xywh", normalized=True):
        _orig_init(self, bboxes, segments, keypoints, bbox_format, normalized)
        u = PENDING["uid"]
        if u is not None and len(bboxes) == len(u):
            self.uid = np.asarray(u).copy()

    Instances.__init__ = _init

    # ---- 2) Instances.__getitem__：索引/布尔掩码时同步搬 uid ----
    _orig_getitem = Instances.__getitem__

    def _getitem(self, index):
        r = _orig_getitem(self, index)
        u = getattr(self, "uid", None)
        if u is not None:
            try:
                r.uid = np.asarray(u)[index]
            except Exception:  # noqa: BLE001
                _note_desync("getitem")
        _check(r, "getitem")
        return r

    Instances.__getitem__ = _getitem

    # ---- 3) Instances.concatenate：Mosaic 四拼 ----
    _orig_cat = Instances.concatenate

    def _cat(instances, axis=0):
        r = _orig_cat(instances, axis)
        if all(hasattr(i, "uid") and getattr(i, "uid") is not None for i in instances):
            r.uid = np.concatenate([np.asarray(i.uid) for i in instances], axis=axis)
        else:
            STATS["concat_no_uid"] += 1
        _check(r, "concatenate")
        return r

    Instances.concatenate = staticmethod(_cat)

    # ---- 4) remove_zero_area_boxes：Mosaic._cat_labels 的裁剪过滤 ----
    _orig_rzab = Instances.remove_zero_area_boxes

    def _rzab(self):
        good = _orig_rzab(self)
        u = getattr(self, "uid", None)
        if u is not None:
            if len(u) == len(good):
                self.uid = np.asarray(u)[good]
            else:
                _note_desync("remove_zero_area_boxes")
                self.uid = None
        return good

    Instances.remove_zero_area_boxes = _rzab

    # ---- 5) RandomPerspective.__call__：给内部新建的 Instances 提供 PENDING uid ----
    from ultralytics.data.augment import RandomPerspective

    _orig_rp = RandomPerspective.__call__

    def _rp(self, labels):
        inst = labels.get("instances")
        u = getattr(inst, "uid", None) if inst is not None else None
        if u is None:
            STATS["rp_pending_miss"] += 1
        # RandomPerspective 的**输入** == Mosaic 的输出（CopyPaste p=0 是 no-op），
        # 因此这里是唯一可靠的 "mosaic 之后" 快照点。
        if _ON_STAGE["fn"] is not None:
            _ON_STAGE["fn"]("mosaic", labels, None if u is None else np.asarray(u))
        PENDING["uid"] = None if u is None else np.asarray(u)
        try:
            r = _orig_rp(self, labels)
        finally:
            PENDING["uid"] = None
        ri = r.get("instances")
        if ri is not None:
            _check(ri, "RandomPerspective")
            if getattr(ri, "uid", None) is None and u is not None:
                _note_desync("rp_uid_lost")
        return r

    RandomPerspective.__call__ = _rp

    # ---- 6) get_image_and_label：给每个源图（主图 + 3 个 mosaic tile）打 uid ----
    from ultralytics.data.base import BaseDataset

    _orig_gial = BaseDataset.get_image_and_label

    def _gial(self, index):
        lab = _orig_gial(self, index)
        inst = lab.get("instances")
        if inst is not None:
            n = len(inst)
            inst.uid = np.array([encode(index, k) for k in range(n)], dtype=np.int64)
            if _ON_STAGE["fn"] is not None:
                _ON_STAGE["fn"]("native", lab, inst.uid)
        return lab

    BaseDataset.get_image_and_label = _gial

    # ---- 7) Format.__call__：final（它 pop 掉 instances，所以必须在其内部前抓）----
    from ultralytics.data.augment import Format

    _orig_fmt = Format.__call__

    def _fmt(self, labels):
        inst = labels.get("instances")
        u = getattr(inst, "uid", None) if inst is not None else None
        if u is not None and _ON_STAGE["fn"] is not None:
            _ON_STAGE["fn"]("final", labels, np.asarray(u))
        r = _orig_fmt(self, labels)
        return r

    Format.__call__ = _fmt
    return PENDING
