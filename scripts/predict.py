"""scripts/predict.py — 比赛最终提交预测文件生成（推理）。

职责：加载训练好的 best.pt（默认 F4 YOLO11m RGBD 中期融合模型，lr0=0.005，imgsz=1280）→ 在 test 集上推理
→ 每张图生成同名 TXT（YOLO 归一化格式，含置信度）→ 打包成 submission.zip。

设计要点（对应需求逐条落实）:
  - 输入: data/raw/test/{visible, infrared, depth}/，三模态同名、空间对齐；
    RGBD 模型只读 visible + depth（4 通道 [B,G,R,D]）。
  - 输出格式: 每行 `class_id cx cy w h conf`，6 个值空格分隔，坐标为归一化
    [0,1]，保留 6 位小数，class_id ∈ [0,11]。
  - 边界情况: 无检测 → 空 TXT（文件必须存在）；每张图最多 100 个框（按 conf
    降序）；坐标 clamp 到 [0,1]；过滤 class_id 越界。
  - conf_thres 默认 0.001（NMS 前过滤），NMS IoU 默认 0.7（与训练 val 一致）。
  - 多模态参数 use_simotm / pairs_rgb_ir / pairs_rgb_depth / channels 从 train
    配置显式读取并注入加载器（checkpoint 的 _reset_ckpt_args 会丢弃这些字段，
    与 evaluate.py 同理），不依赖 checkpoint 内残留 args。
  - TTA（可选）: 原图 + 水平翻转两次推理，翻转结果水平还原后按 IoU=0.7 做
    逐类 NMS 合并。
  - 设备: auto → GPU 可用则 GPU，否则 CPU。
  - 输出目录: submissions/predict_YYYYMMDD_HHMMSS/results/*.txt + submission.zip
  - --debug: 只处理前 10 张图并打印详细信息。

用法:
  python scripts/predict.py                                    # 默认 F4 best.pt
  python scripts/predict.py --weights runs/.../best.pt --tta
  python scripts/predict.py --conf 0.001 --iou 0.7 --batch 16
  python scripts/predict.py --use_simotm RGBD --pairs_rgb_ir visible,depth \
      --pairs_rgb_depth visible,depth --channels 4

说明:
  - 本脚本走「自包含推理循环」：直接用 fork 的 LoadImagesAndVideos 读多模态图，
    用 DetectionModel 前向 + non_max_suppression + scale_boxes 得到检测框，再自己
    写 TXT。不经过 model.predict()/model.val()（它们会受 checkpoint args 残留影响，
    且 predictor 的 setup_source 未转发 pairs_rgb_depth），从而对多模态参数和 TTA
    有完全控制，且对 ultralytics 代码零侵入。
"""
from __future__ import annotations

import argparse
import datetime
import os
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml

# 保证 `python scripts/predict.py` 时能导入项目内的 ultralytics / scripts
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.data.loaders import LoadImagesAndVideos  # noqa: E402
from ultralytics.utils.ops import non_max_suppression, scale_boxes  # noqa: E402

try:
    from torchvision.ops import nms as torch_nms
except Exception:  # pragma: no cover - 极老 torchvision 兜底
    torch_nms = None

# F4 默认权重 / 训练配置 / 测试输入（YOLO11m RGBD，imgsz=1280，lr0=0.005，best mAP50-95=0.53273）
DEFAULT_WEIGHTS = "runs/urban_multimodal_det_yolo11_rgbd_f4_1280/weights/best.pt"
DEFAULT_TRAIN_CONFIG = "configs/train_f4_1280.yaml"
DEFAULT_SOURCE = "data/raw/test/visible"

CLASS_ID_MIN, CLASS_ID_MAX = 0, 11  # class_id ∈ [0, 11]


# ============================================================
# 基础工具
# ============================================================

def _load_yaml(path) -> dict:
    """读取 yaml 配置文件，返回解析后的字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_pairs(s: str):
    """把 "visible,depth" 解析为 ['visible', 'depth']。"""
    if s is None:
        return None
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"pairs 参数需形如 'a,b'（两个非空 token），收到: {s!r}")
    return parts


def _resolve_device(arg: str):
    """把 --device 解析为 torch device 与显示字符串。

    auto → GPU 可用则 GPU，否则 CPU；cpu → CPU；其余（如 "0"/"0,1"）→ CUDA。
    """
    arg = str(arg).lower()
    if arg in ("auto", ""):
        use_cuda = torch.cuda.is_available()
    elif arg == "cpu":
        use_cuda = False
    else:
        use_cuda = torch.cuda.is_available()  # "0" / "0,1" 视为 cuda
    return torch.device("cuda" if use_cuda else "cpu")


# ============================================================
# 图像 → 张量（通道顺序对齐 fork 的 BasePredictor.preprocess）
# ============================================================

def _to_chw(im: np.ndarray) -> np.ndarray:
    """把 HWC uint8 图转成 CHW uint8 数组，通道顺序对齐 fork 的 preprocess。

    输入 im 为 BGR 主图 + 额外模态通道的合并图：
      RGBD(4ch) [B,G,R,D]   → [R,G,B,D]
      RGBID(5ch)[B,G,R,IR,D]→ [R,G,B,IR,D]（fork 的 5ch 分支会错误地整体反转，
                             这里按训练 Format 的正确语义实现）
      其余(1/3/6ch) 也做对应处理。
    """
    im = np.ascontiguousarray(im.transpose(2, 0, 1))  # CHW
    c = im.shape[0]
    if c == 1:
        pass  # 单通道不动
    elif c == 3:
        im = im[::-1]  # BGR -> RGB
    elif c in (4, 5):
        # 前 3 通道 BGR→RGB，其余模态通道保持原序 → [R,G,B,...]
        im = np.concatenate([im[:3][::-1], im[3:]], axis=0)
    elif c == 6:
        im = np.concatenate([im[:3][::-1], im[3:][::-1]], axis=0)
    else:  # 多光谱等，退化为整体反转
        im = im[::-1]
    return np.ascontiguousarray(im)


def _preprocess_batch(imgs, letterbox: LetterBox, device: torch.device):
    """对一批 BGR(+多模态) HWC uint8 图做 letterbox + CHW 转换，返回 (张量, 原图尺寸列表)。"""
    orig_shapes = [im.shape[:2] for im in imgs]  # 每张 (H, W)
    arrs = [_to_chw(letterbox(image=im)) for im in imgs]
    x = torch.from_numpy(np.stack(arrs)).to(device).float() / 255.0
    return x, orig_shapes


# ============================================================
# TTA 合并（逐类 NMS）
# ============================================================

def _nms_merge(boxes: torch.Tensor, iou_thres: float) -> torch.Tensor:
    """对已解码的 [N,6]=xyxy,conf,cls 做逐类 NMS 合并（TTA 原图+翻转去重）。

    逐类 NMS 避免把不同类但重叠的框误合并；同类重叠框（同一目标的原图/翻转双检测）
    按 IoU 阈值去重。
    """
    if len(boxes) <= 1:
        return boxes
    if torch_nms is None:
        return boxes  # 无 torchvision NMS 时退化为不合并（保留全部）
    keep_list = []
    for c in boxes[:, 5].unique():
        m = boxes[:, 5] == c
        keep = torch_nms(boxes[m, :4], boxes[m, 4], iou_thres)
        keep_list.append(boxes[m][keep])
    return torch.cat(keep_list, dim=0) if keep_list else boxes[:0]


# ============================================================
# 检测框 → 提交文本行
# ============================================================

def _format_lines(dets, orig_h: int, orig_w: int, max_boxes: int):
    """把原图坐标系的检测框 [N,6]=xyxy,conf,cls 转成提交文本行（字符串列表）。

    处理: 按 conf 降序、截断到 max_boxes、过滤 class_id 越界、归一化 + clamp [0,1]、
    6 位小数。无检测时返回空列表（调用方仍会写出空文件）。
    """
    lines = []
    if dets is None or len(dets) == 0:
        return lines
    d = dets.detach().cpu().numpy()
    d = d[np.argsort(-d[:, 4])]  # conf 降序
    d = d[:max_boxes]
    for x1, y1, x2, y2, conf, cls in d:
        cid = int(round(float(cls)))
        if cid < CLASS_ID_MIN or cid > CLASS_ID_MAX:
            continue
        cx = (x1 + x2) / 2.0 / orig_w
        cy = (y1 + y2) / 2.0 / orig_h
        w = (x2 - x1) / orig_w
        h = (y2 - y1) / orig_h
        # clamp 到 [0,1]（scale_boxes 已 clip 到像素框内，这里是浮点防御）
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}")
    return lines


# ============================================================
# 校验输出
# ============================================================

def _check_overflow(results_dir: Path) -> tuple[int, int]:
    """检查 results 下所有 TXT：坐标是否越界 [0,1]、class_id 是否越界。

    返回 (越界行数, 检查行总数)。
    """
    bad, total = 0, 0
    for txt in sorted(results_dir.glob("*.txt")):
        for line in txt.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 6:
                bad += 1
                continue
            total += 1
            try:
                cid = int(parts[0])
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                bad += 1
                continue
            if cid < CLASS_ID_MIN or cid > CLASS_ID_MAX:
                bad += 1
            if any(v < 0.0 or v > 1.0 for v in vals):
                bad += 1
    return bad, total


def _validate(results_dir: Path, zip_path: Path, n_test: int):
    """打印输出自检：数量、空文件、随机抽样、坐标越界、zip 大小/数量。"""
    txts = sorted(results_dir.glob("*.txt"))
    empty = [t for t in txts if t.stat().st_size == 0]
    bad, total = _check_overflow(results_dir)

    print("\n" + "=" * 60)
    print("输出校验")
    print("=" * 60)
    print(f"生成 TXT 数量   : {len(txts)} / test 图像 {n_test} "
          f"{'✓' if len(txts) == n_test else '✗ 数量不符'}")
    print(f"空 TXT 数量     : {len(empty)}（无检测的图像）")
    print(f"坐标/类别越界   : {bad} 行异常 / 共 {total} 行检测")

    if txts:
        rng = random.Random(42)
        sample = rng.sample(txts, min(3, len(txts)))
        print("-" * 60)
        print("随机抽样（前 5 行）:")
        for t in sample:
            head = t.read_text(encoding="utf-8").splitlines()[:5]
            print(f"  [{t.name}] " + ("(空)" if not head else ""))
            for ln in head:
                print(f"      {ln}")

    if zip_path is not None and zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print("-" * 60)
        print(f"submission.zip  : {zip_path}")
        print(f"  zip 大小      : {size_mb:.2f} MB")
        print(f"  zip 内文件数  : {len(names)}")
    print("=" * 60 + "\n")


# ============================================================
# 主流程
# ============================================================

def _parse_args():
    parser = argparse.ArgumentParser(description="比赛提交预测文件生成（推理）")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                        help=f"权重文件路径，默认 {DEFAULT_WEIGHTS}")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE,
                        help=f"测试 visible 主输入目录，默认 {DEFAULT_SOURCE}")
    parser.add_argument("--train_config", type=str, default=DEFAULT_TRAIN_CONFIG,
                        help=f"训练配置（用于读取多模态参数），默认 {DEFAULT_TRAIN_CONFIG}")
    parser.add_argument("--use_simotm", type=str, default=None,
                        help="多模态模式 RGBD/RGBID/Depth/Infrared/RGBT；默认取 train_config")
    parser.add_argument("--pairs_rgb_ir", type=str, default=None,
                        help="第二模态映射 'a,b'；默认取 train_config")
    parser.add_argument("--pairs_rgb_depth", type=str, default=None,
                        help="深度第三模态映射 'a,b'（RGBID 用）；默认取 train_config")
    parser.add_argument("--channels", type=int, default=None,
                        help="模型输入通道数；默认取 train_config")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="置信度阈值（NMS 前过滤），默认 0.001")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU 阈值，默认 0.7（与训练 val 一致）")
    parser.add_argument("--imgsz", type=int, default=1280, help="推理输入尺寸，默认 1280（与训练一致）")
    parser.add_argument("--batch", type=int, default=16, help="推理批大小，默认 16")
    parser.add_argument("--max_det", type=int, default=300,
                        help="NMS 每图最大检测数，默认 300")
    parser.add_argument("--max_boxes", type=int, default=100,
                        help="每图最终写入的最大框数，默认 100")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto | cpu | 0 | 0,1；默认 auto")
    parser.add_argument("--output_dir", type=str, default="submissions",
                        help="输出根目录，默认 submissions")
    parser.add_argument("--tta", action="store_true",
                        help="启用 TTA（原图 + 水平翻转，逐类 NMS IoU=0.7 合并）")
    parser.add_argument("--debug", action="store_true",
                        help="调试模式：只处理前 10 张图并打印详细信息")
    return parser.parse_args()


def main():
    args = _parse_args()

    # ---- 权重 ----
    weights_path = (PROJECT_ROOT / args.weights).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    # ---- 多模态参数：CLI 优先，否则从 train 配置读取（checkpoint 不残留这些字段） ----
    train_cfg = _load_yaml(PROJECT_ROOT / args.train_config) \
        if (PROJECT_ROOT / args.train_config).exists() else {}
    use_simotm = args.use_simotm or str(train_cfg.get("use_simotm", "RGBD"))
    pairs_rgb_ir = (_parse_pairs(args.pairs_rgb_ir)
                    if args.pairs_rgb_ir is not None
                    else list(train_cfg.get("pairs_rgb_ir", ["visible", "depth"])))
    pairs_rgb_depth = (_parse_pairs(args.pairs_rgb_depth)
                       if args.pairs_rgb_depth is not None
                       else list(train_cfg.get("pairs_rgb_depth", ["visible", "depth"])))
    channels = int(args.channels if args.channels is not None
                   else train_cfg.get("channels", 4))

    device = _resolve_device(args.device)
    source_dir = (PROJECT_ROOT / args.source).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"测试输入目录不存在: {source_dir}")

    # ---- 加载模型 ----
    print(f"[predict] weights={weights_path}")
    print(f"[predict] source={source_dir}")
    print(f"[predict] use_simotm={use_simotm} pairs_rgb_ir={pairs_rgb_ir} "
          f"pairs_rgb_depth={pairs_rgb_depth} channels={channels}")
    print(f"[predict] device={device} imgsz={args.imgsz} batch={args.batch} "
          f"conf={args.conf} iou={args.iou} tta={args.tta}")

    model = YOLO(str(weights_path))
    nn_model = model.model  # DetectionModel (nn.Module)
    nn_model.to(device).eval()
    nc = int(getattr(nn_model, "nc", len(model.names) or 12))
    if nc != 12:
        print(f"[warn] 模型 nc={nc}（预期 12），请确认类别数与比赛一致")
    model_ch = int(nn_model.yaml.get("ch", channels))
    if model_ch != channels:
        print(f"[warn] --channels={channels} 与模型 ch={model_ch} 不一致，"
              f"以模型为准（加载器按 --use_simotm 决定实际通道数）")
    stride = int(nn_model.stride.max()) if getattr(nn_model, "stride", None) is not None else 32

    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False,
                          scaleFill=False, scaleup=True, center=True, stride=stride)

    # ---- 输出目录 ----
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = PROJECT_ROOT / args.output_dir / f"predict_{stamp}"
    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- 数据加载器（多模态读取交给 fork 的 LoadImagesAndVideos） ----
    loader = LoadImagesAndVideos(
        str(source_dir),
        batch=args.batch,
        use_simotm=use_simotm,
        imgsz=args.imgsz,
        pairs_rgb_ir=pairs_rgb_ir,
        pairs_rgb_depth=pairs_rgb_depth,
    )
    n_test = loader.ni  # 测试图像总数（不含视频）

    # ---- 推理主循环 ----
    processed = 0
    n_truncated = 0  # NMS 后单图框数超过 max_boxes 的图片数（用于判断 100 框截断是否影响验证）
    max_seen = 0     # 所有图片中单图最大框数
    limit = 10 if args.debug else float("inf")
    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_test if not args.debug else min(n_test, 10), desc="infer", unit="img")
    except Exception:
        pbar = None

    for paths, imgs, _info in loader:
        if processed >= limit:
            break

        x, orig_shapes = _preprocess_batch(imgs, letterbox, device)

        # 原图推理 + NMS
        with torch.no_grad():
            preds = nn_model(x)
        out = non_max_suppression(preds, args.conf, args.iou, nc=nc, max_det=args.max_det, multi_label=True)
        for i in range(len(out)):
            if out[i] is not None and len(out[i]):
                out[i][:, :4] = scale_boxes(x.shape[2:], out[i][:, :4], orig_shapes[i])

        # TTA：水平翻转二次推理，还原后逐类 NMS 合并
        if args.tta:
            xf = torch.flip(x, dims=[3])  # 宽度维水平翻转
            with torch.no_grad():
                preds_f = nn_model(xf)
            out_f = non_max_suppression(preds_f, args.conf, args.iou, nc=nc, max_det=args.max_det, multi_label=True)
            W_lb = x.shape[3]
            for i in range(len(out)):
                df = out_f[i]
                if df is None or len(df) == 0:
                    continue
                df = df.clone()
                df[:, [0, 2]] = W_lb - df[:, [2, 0]]  # 水平还原 x 坐标
                df[:, :4] = scale_boxes(x.shape[2:], df[:, :4], orig_shapes[i])
                dn = out[i]
                merged = torch.cat([dn, df], dim=0) if (dn is not None and len(dn)) else df
                out[i] = _nms_merge(merged, args.iou)

        # 写 TXT
        for i, p in enumerate(paths):
            if processed >= limit:
                break
            orig_h, orig_w = orig_shapes[i]
            n_dets = 0 if out[i] is None else len(out[i])
            max_seen = max(max_seen, n_dets)
            if n_dets > args.max_boxes:
                n_truncated += 1
            lines = _format_lines(out[i], orig_h, orig_w, args.max_boxes)
            out_txt = results_dir / (Path(p).stem + ".txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.writelines("\n".join(lines) + ("\n" if lines else ""))
            if args.debug:
                print(f"[debug] {Path(p).name}: {len(lines)} 框")
            processed += 1
            if pbar is not None:
                pbar.update(1)
        if processed >= limit:
            break

    if pbar is not None:
        pbar.close()

    # ---- 打包 zip（archive 内为 results/<name>.txt） ----
    zip_path = out_root / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for txt in sorted(results_dir.glob("*.txt")):
            zf.write(txt, arcname=f"results/{txt.name}")

    print(f"[predict] 完成：{processed} 张图 → {results_dir}")
    print(f"[predict] 框数统计: 单图最大框数={max_seen}, "
          f"超过 max_boxes({args.max_boxes}) 被截断的图片数={n_truncated}")
    print(f"[predict] 打包   → {zip_path}")

    # ---- 校验 ----
    _validate(results_dir, zip_path, n_test)


if __name__ == "__main__":
    main()
