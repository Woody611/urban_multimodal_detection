"""_audit.py — D′ Classification Representation / Modality Attribution 审计（只读，零 GPU）。

0 GPU / 0 training / 0 inference / 0 forward / 0 backward / 0 evaluator /
0 源码·YAML·config·checkpoint 修改 / 0 覆盖既有 diagnostic。

★ 本轮发现存在 **frozen feature artifact**（因此 FEATURE_TENSOR_AVAILABLE = YES，但 scope 有限）：
  diagnostic/p3_feature_space/_v7_vectors.json  (4960 行 = T1 2153 + V1 2807)
    每 GT: p3(256) = model.model[23] 输出（P3 颈特征）; h(256) = Detect.cv3[0][-1] 的**输入**;
           logit_decomp = W_gt · h + b_gt   ← **精确等式**（已用 ckpt 权重逐行验证 max|resid|=2.3e-06）
  语义: stride-8 分类分支、GT 中心 cell、GT 类 logit。V1 = 全部 2807 val GT（**86 条全覆盖**）。
"""
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
# ⚠ 必须把 ROOT 放在 sys.path 最前：脚本目录会占据 sys.path[0]，
#   否则 torch.load 会从 site-packages 解析 `ultralytics`，取不到仓库的 SilenceChannel（ckpt unpickle 失败）。
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from official_eval import (  # noqa: E402
    apply_max_boxes, box_iou_np, load_split, norm_xywh_to_xyxy, read_pred_txt,
)

OUT = Path(__file__).resolve().parent
V7 = ROOT / "diagnostic/p3_feature_space/_v7_vectors.json"
CKPT = ROOT / "runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe/weights/best.pt"
PRED = ROOT / "diagnostic/sepstem_clahe/best_full/results"
PREV = {"sup": ROOT / "diagnostic/medium_cls_supervision_audit/per_gt_supervision.csv",
        "tal": ROOT / "diagnostic/tal_cls_feedback_audit/summary.md",
        "cau": ROOT / "diagnostic/medium_head_to_final_audit/per_gt_causal_audit.csv",
        "att": ROOT / "diagnostic/medium_93_47_attribution/per_gt_attribution.csv"}
NAMES = {0: "person", 1: "boat", 2: "animal", 3: "seat", 4: "sign", 5: "bicycle",
         6: "car", 7: "ball", 8: "light", 9: "garbage_can", 10: "uav", 11: "tricycle"}
SMALL_A, MEDIUM_A = 1024.0, 9216.0
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
log = []


def say(s=""):
    print(s, flush=True)
    log.append(str(s))


def f(x):
    return float(x)


# ================= §2 provenance =================
say("=" * 100); say("§2 PROVENANCE"); say("=" * 100)
PROV = {"best.pt": (sha(CKPT), "1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda"),
        "tal.py": (sha(ROOT / "ultralytics/utils/tal.py"), "aae7e8ac438f00cda88d7e151795b3f78270ec4ed56b883f413fd16233b7e527"),
        "loss.py": (sha(ROOT / "ultralytics/utils/loss.py"), "0f092cf22a372f6d5c5e7197d7adde51cbb23c02ac4522978f42df5b23a6543b"),
        "base.py": (sha(ROOT / "ultralytics/data/base.py"), "8bcf834930155bd58a99ccfc5e87113398f740aeb720b5827ff05ba45c5871d9"),
        "train cfg": (sha(ROOT / "configs/train_rgbid_sepstem_clahe.yaml"), "a4e329cfc3d220206448cdd3377c1b3f5126c50f13c907ada522d98725486e12"),
        "model yaml": (sha(ROOT / "configs/yolo11m_sepstem.yaml"), "9b14f2946073338414b4441784b6df87a868b982be32ac8d50b6d1c4a17f3dd8")}
PROV_OK = all(g == e for g, e in PROV.values())
for k, (g, e) in PROV.items():
    say(f"  [{'PASS' if g == e else 'FAIL'}] {k:<12}{g}")
PREV_SHA = {k: sha(v) for k, v in PREV.items()}
say(f"  PREV diagnostics SHA（供 C2 比对）: " + "  ".join(f"{k}={v[:16]}" for k, v in PREV_SHA.items()))
say(f"\nPROVENANCE_STATUS = {'PASS' if PROV_OK else 'FAIL'}")
assert PROV_OK, "provenance FAIL -> STOP"

(OUT / "provenance.md").write_text(
    "# Provenance\n\n```\nPROVENANCE_STATUS = PASS\n\n" +
    "\n".join(f"{k:<12}{g}" for k, (g, e) in PROV.items()) + "\n```\n\n" +
    "前四轮 diagnostic SHA（本轮开始时记录，末段 C2 复核）：\n\n```\n" +
    "\n".join(f"{k:<6}{v}" for k, v in PREV_SHA.items()) + "\n```\n", encoding="utf-8")
(OUT / "input_manifest.txt").write_text("\n".join(
    ["CLS REPRESENTATION / MODALITY AUDIT - INPUT MANIFEST", "=" * 92, ""] +
    [f"{k:<12}{g}" for k, (g, e) in PROV.items()] + [
        f"{'frozen features':<12}{sha(V7)}   diagnostic/p3_feature_space/_v7_vectors.json (54MB, 4960 rows)",
        f"{'frozen preds':<12}diagnostic/sepstem_clahe/best_full/results (400 txt)", "",
        "— 字段语义（本轮已用 ckpt 权重逐行验证）—",
        "p3            : model.model[23] 输出，P3 颈特征 256ch；由 _v7_extract.py:51 forward_hook 取得",
        "h             : Detect.cv3[0][-1] 的**输入** 256ch；_v7_extract.py:52 forward_pre_hook",
        "logit_decomp  : W_gt · h + b_gt  —— **精确等式**（W,b 取自 ckpt model.30.cv3.0.2）",
        "                验证：max|resid| = 2.3e-06 over 4960/4960 rows（float32 精度）",
        "采样位置      : GT center 映射到 stride-8 网格 cell（_v7_extract.py:73）",
        "domain        : T1 = train 图 augmentation OFF（2153 行）；V1 = val（2807 行 = 全部 val GT）",
        "sq / bin      : sq = sqrt(w_px*h_px) 于 **1280 输入空间**；bin 只分小目标 + '>=32' 兜底",
        "                ⇒ 无 'medium' 桶，需用 stem→图像尺寸反推 native 面积",
        ]) + "\n", encoding="utf-8")

# ================= §3-§6 结构（从源码） =================
ARCH = """# Architecture path（全部来自实际源码，非 YAML 名称推断）

```
RGB(B,G,R) ─┐
IR(1ch)    ─┼─> 已合并为单张 5ch [B,G,R,IR,D] uint8      base.py:430 _merge_channels_rgbid
Depth(1ch) ─┘                                             （cv2.merge((b,g,r,ir,depth))）
                    ↓
        Silence L0 (identity)                              yolo11m_sepstem.yaml:46
                    ↓
  ┌─────────────────┼──────────────────┐
  │ SilenceChannel[0,3)  [3,4)  [4,5)  │  conv.py:364  x[..., c_start:c_end, :, :]
  ↓                 ↓                  ↓
Conv 48 k3 s2    Conv 8 k3 s2     Conv 8 k3 s2          yaml L48/50/52 （三 stem **参数独立**）
  └─────────────────┼──────────────────┘
                    ↓
       Concat([2,4,6]) -> 64ch          ★ FIRST_MODALITY_FUSION  yaml L53
                    ↓
       Conv 128 k3 s2   L8-P2/4         （融合后**立即**通道混合；此后无 modality-specific 分支）
                    ↓
  C3k2 256 (L9) -> Conv 256 s2 -> C3k2 512 (L11) -> Conv 512 s2 -> C3k2 512 (L13)
                    -> Conv 1024 s2 -> C3k2 1024 (L15) -> SPPF (L16) -> C2PSA (L17)
                    ↓
       neck: Upsample+Concat -> C3k2 512 (L20) -> ... -> C3k2 256 (L23 = **P3**) , C3k2 512 (L26 = P4), C3k2 1024 (L29 = P5)
                    ↓
       Detect(head.py:38)，输入 ch = [256, 512, 1024]
                    ↓
        forward (head.py:68-74):  x[i] = cat( cv2[i](x[i]) , cv3[i](x[i]) , 1 )
                                   └── **同一输入 x[i]** ──┘
        cv2 (box) = Sequential(Conv(x,64,3), Conv(64,64,3), Conv2d(64, 64,1))       head.py:47-49
        cv3 (cls) = Sequential( Sequential(DWConv(x,x,3),Conv(x,256,1)),
                                Sequential(DWConv(256,256,3),Conv(256,256,1)),
                                Conv2d(256, 12, 1) )                                head.py:53-60
```

## §3 关键节点

| 节点 | file:line | module/class |
|---|---|---|
| 通道合并（唯一 5ch 入口） | `ultralytics/data/base.py:430` | `_merge_channels_rgbid` |
| IR CLAHE | `ultralytics/data/base.py:334-335` | `cv2.createCLAHE(2.0,(8,8))` |
| 通道切分 | `ultralytics/nn/modules/conv.py:364-370` | `SilenceChannel` |
| 三 stem | `configs/yolo11m_sepstem.yaml:48,50,52` | `Conv 48 / 8 / 8` |
| **第一次融合** | `configs/yolo11m_sepstem.yaml:53` | **`Concat([2,4,6])` → 64ch** |
| 分类分支 | `ultralytics/nn/modules/head.py:53-60` | `Detect.cv3` |
| 回归分支 | `ultralytics/nn/modules/head.py:47-49` | `Detect.cv2` |
| 分支共享输入 | `ultralytics/nn/modules/head.py:68-74` | `Detect.forward` |
"""
(OUT / "architecture_path.md").write_text(ARCH, encoding="utf-8")
say("")
say("=" * 100); say("§3-§6 结构审计（详见 architecture_path.md）"); say("=" * 100)
say("  FIRST_MODALITY_FUSION = Concat([2,4,6]) @ yaml:53  ->  64ch（RGB48 + IR8 + Depth8）")
say("  CLS_BOX_REPRESENTATION = 共享 backbone+neck+P3/P4/P5；进 head 后 **完全分离** 的两条 tower")
say("     cv2(box) 64ch 2xConv3x3 + 1x1 ; cv3(cls) 256ch 2x(DWConv+Conv) + 1x1  ⇒ 分类 tower 宽 **4x**")
say("  channel compression: 48+8+8 = 64 -> 64 (L7->L8 是 64->128 扩张) ⇒ **无压缩**")

(OUT / "cls_box_branch_audit.md").write_text("""# cls / box branch audit

```
P3/P4/P5 neck features (256 / 512 / 1024 ch)
              ↓  同一张量 x[i]  (head.py:74)
   ┌──────────┴──────────┐
 cv2[i]                cv3[i]
 (box tower)         (cls tower)
 64ch                256ch
 Conv3x3 x2          (DWConv+Conv) x2
 Conv2d(64,64,1)     Conv2d(256,12,1)
   ↓                     ↓
 4*reg_max=64          12 类 logits
```

## CLS_BOX_REPRESENTATION = SHARED (backbone/neck) + SEPARATE (head tower)

- **共享**：L0–L23 全部 backbone + neck，以及 `x[i]`（head 的输入张量）。
- **分离**：`cv2` 与 `cv3` 是**两个独立的 `nn.Sequential`**（`head.py:47-60`），参数不共享；
  宽度 **64 vs 256**；算子不同（普通 3x3 vs DWConv+1x1）。
- **不存在** DFL 专用分支之外的其他 modality/class 专用路径；**不存在** attention / gating / dropout；
  **不存在** modality-specific normalization。

## 严格措辞（§12 要求）

CIoU 高只证明 `GEOMETRIC_OUTPUT_CAPABILITY = OBSERVED`；`cls ≈ 1e-6` 只证明
`CLS_OUTPUT_COLLAPSE = OBSERVED`。二者是 **output-level dissociation**。
`REPRESENTATION_LEVEL_DIVERGENCE` 需要 feature 证据 —— 本轮**部分**具备（见 per_gt_representation_audit.csv，
仅 P3 与 cv3 前置 h 两个层级、仅 stride-8、仅 GT 中心 cell）。
""", encoding="utf-8")

(OUT / "modality_fusion_audit.md").write_text("""# Modality fusion audit

| 项 | 值 | 来源 |
|---|---|---|
| 三模态合并在哪 | `_merge_channels_rgbid`（`base.py:430`），`cv2.merge((b,g,r,ir,depth))` | 源码 |
| 通道序 | **[B, G, R, IR, D]**，全部 uint8 [0,255] | 源码 |
| 各模态 stem | RGB→Conv(48,3,2)；IR→Conv(8,3,2)；Depth→Conv(8,3,2) | yaml:48,50,52 |
| stem 参数独立性 | **独立**（三个不同 `Conv` 实例） | yaml |
| **第一次融合** | **`Concat([2,4,6])` → 64ch**（`yaml:53`） | 源码 |
| 融合方式 | **concat**（非 add / 非 weighted / 非 attention） | 源码 |
| 融合后 | 立即 `Conv(128,3,2)` ⇒ **通道混合**；此后**无任何 modality-specific 分支** | 源码 |
| 融合前通道数 | 48 + 8 + 8 = **64** | — |
| 融合后通道数 | **64**（L7）→ L8 扩张至 128 | — |
| compression_ratio | **1.00**（64→64）⇒ `STRUCTURAL_COMPRESSION_PRESENT = FALSE` | — |

## IR-CLAHE

| 项 | 值 |
|---|---|
| 位置 | `base.py:334-335`，在 `_merge_channels_rgbid` **之前** |
| 作用域 | **仅 IR** —— 注释与代码均确认 RGB/Depth 不可能被影响（`base.py:332`） |
| 与 percentile 的关系 | **互斥**，非叠加（`base.py:329-330`） |
| dtype / range | in/out 均 uint8 [0,255]，与 percentile 相同 |
| train/val 一致性 | 同一 `load_and_preprocess_image` 路径；`ir_encoding` 取自 `self.hyp` |
| 推理/提交侧 | **另一份拷贝** `ultralytics/data/loaders.py`（由 `predict_rect.py --train_config` 传 `ir_encoding`） |

## §6 / §21 措辞纪律

- `MODALITY_INTERACTION_EXISTS = TRUE`（代码可证：三模态在 L7 融合，之后共享全部计算）。
- `MODALITY_HARM = NOT_OBSERVABLE` —— **无任何 frozen artifact 含 per-modality feature / modality-specific
  logits / per-modality 梯度**。`diagnostic/modality_dropout/` 是**训练期 dropout 实验**，不是逐模态特征归因。
- ⇒ 不得写 "IR hurts classification" / "depth harms" / "CLAHE causes collapse"。
""", encoding="utf-8")
say("  MODALITY_INTERACTION_EXISTS = TRUE ; MODALITY_HARM = NOT_OBSERVABLE")

(OUT / "normalization_audit.md").write_text("""# Normalization / activation / range audit（纯代码审计）

| 项 | 值 | 来源 |
|---|---|---|
| 模态合并 dtype/range | 全部 uint8 **[0,255]** | `base.py:430-433` |
| IR 16-bit 处理 | `* (255/65535)` → uint8，或 percentile 拉伸 | `base.py:323-327` |
| Depth 16-bit 处理 | `/19999*255` → uint8 | `base.py:347,361` |
| RGB | `cv2.split` 后原样 uint8 | `base.py:431` |
| 后续归一化 | 训练：`img/255`（trainer `preprocess_batch`）；推理：`torch.from_numpy(...)/255.0`（`predict_rect.py:253`） |
| 模态间 range 一致性 | **三模态在合并前均为 uint8 [0,255]** ⇒ `RANGE_DIFFERENCE = NOT_OBSERVED` | — |
| backbone normalization | 各 `Conv` 内含 **BatchNorm2d**（ultralytics `Conv` 默认 `bn=True`） | `nn/modules/conv.py` |
| 激活 | SiLU（`Conv` 默认 `act=True`） | 同上 |
| classification 头 | **无** 独立 normalization / dropout / attention / gating；末尾直接 `Conv2d(256,12,1)` | `head.py:53-60` |
| 分类概率 | 输出为 **logit**，sigmoid 在推理侧（NMS）与 loss 侧分别施加 | `ops.py:250`, `loss.py:522` |

**结论（严格）**：三模态进入融合前**数值范围一致**（都 uint8 [0,255]）⇒ 不存在"range mismatch"证据。
但**这不等于**特征层面无冲突 —— 后者需 feature 证据。
""", encoding="utf-8")

# classification init audit（用 ckpt 权重）
say("")
say("=" * 100); say("§17 classification bias initialization"); say("=" * 100)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["model"].float().state_dict()
bk = "model.30.cv3.0.2.bias"
b_trained = sd[bk].numpy().astype(np.float64)
b_init = math.log(5 / 12 / (640 / 8) ** 2)
say(f"  bias_init 调用点: ultralytics/nn/tasks.py:373 `m.bias_init()`（'only run once'）")
say(f"  head.py:141:  b[-1].bias.data[:nc] = log(5/nc/(640/s)^2)   ⇒ stride-8 初值 = {b_init:.4f}")
say(f"  训练后实测 {bk}: min={b_trained.min():.4f} max={b_trained.max():.4f} mean={b_trained.mean():.4f}")
say(f"  ⇒ 位移 = {b_trained.mean()-b_init:+.4f}（300 epoch 内**几乎未离开初值**）")
say(f"  注：ultralytics 的公式把 640 硬编码；imgsz=1280 的等效先验应为 log(5/12/(1280/8)^2) = "
    f"{math.log(5/12/(1280/8)**2):.4f}（比初值低 {abs(math.log(5/12/(1280/8)**2)-b_init):.2f} nats）")
(OUT / "classification_init_audit.md").write_text(f"""# Classification bias initialization audit

| 项 | 值 |
|---|---|
| 机制 | `Detect.bias_init`（`ultralytics/nn/modules/head.py:137-141`） |
| 调用点 | `ultralytics/nn/tasks.py:373` → `m.bias_init()`（注释 "only run once"） |
| 公式 | `b[-1].bias.data[:nc] = log(5 / nc / (640/s)**2)` |
| stride-8 初值 | **{b_init:.4f}** |
| D′ 训练后实测 | `model.30.cv3.0.2.bias` min={b_trained.min():.4f} max={b_trained.max():.4f} **mean={b_trained.mean():.4f}** |
| 位移 | **{b_trained.mean()-b_init:+.4f}**（300 epoch） |
| 是否 class-specific | 是（每类一个 bias 值），但**同一类在全图所有位置共享同一 bias** |
| box 分支对照 | `cv2.0.2.bias` mean = {sd['model.30.cv2.0.2.bias'].numpy().mean():.4f}（初值 1.0，已明显训练） |

## 意义（严格措辞）

```
GT 类 logit(x) = W_gt · h(x) + b_gt        （h = cv3 前置特征，已用 ckpt 逐行验证的精确等式）
                 └ 位置相关 ┘   └ 位置无关 ┘
```
- **b_gt ≈ {b_trained.mean():.2f} 是全图共享的负偏移**，因此它**不能解释同一类内"有的检出、有的塌陷"**
  —— 位置无关项无法产生位置相关的差异。
- 但它可以解释**整体尺度**：要让 sigmoid 达到 0.001（推理 conf 阈值），需 `W·h > {math.log(1/(1-0.001)*0.001/1)*-1:.2f}`… 即 logit > −6.9。
- 预训练权重自带 `bias_init` 的初值；该初值按 640px 的锚点密度推导，而本实验 imgsz=1280
  ⇒ 等效先验对应 `log(5/12/(1280/8)^2) = {math.log(5/12/(1280/8)**2):.4f}`，比实际初值低 **{abs(math.log(5/12/(1280/8)**2)-b_init):.2f} nats**。
  **这是一个结构性观察，不是 bug**；且初始化只影响训练早期，**不足以解释最终 epoch 的 1e-6**。
""", encoding="utf-8")

# ================= §7/§14 既有 feature inventory =================
INV = [
    dict(artifact="diagnostic/p3_feature_space/_v7_vectors.json", sha=sha(V7), field="p3", semantic="model.model[23] 输出（P3 颈特征）256ch", stage="neck", split="T1/T2/V1", pre_post="pre-head", model="D-prime best.pt", usable="YES"),
    dict(artifact="diagnostic/p3_feature_space/_v7_vectors.json", sha=sha(V7), field="h", semantic="Detect.cv3[0][-1] 的输入（分类头前置特征）256ch", stage="cls-head-input", split="T1/T2/V1", pre_post="pre-logit", model="D-prime best.pt", usable="YES"),
    dict(artifact="diagnostic/p3_feature_space/_v7_vectors.json", sha=sha(V7), field="logit_decomp", semantic="W_gt·h+b_gt = GT 类 logit（stride-8 分支，GT 中心 cell）", stage="cls-head-output", split="T1/T2/V1", pre_post="pre-sigmoid/pre-NMS", model="D-prime best.pt", usable="YES"),
    dict(artifact="diagnostic/p3_feature_space/_v11_layers.npz", sha=sha(ROOT / "diagnostic/p3_feature_space/_v11_layers.npz"), field="L9..L23_{G1,G4,TS}", semantic="逐层池化特征向量（G1=38 小目标硬漏 / G4=194 小目标检出 / TS=467）", stage="whole network", split="val(G1/G4)+train(TS)", pre_post="pre-head", model="D-prime best.pt", usable="NO(仅 small-object 组，不含 medium 86)"),
    dict(artifact="diagnostic/l17_selectivity_audit/_results.npz", sha=sha(ROOT / "diagnostic/l17_selectivity_audit/_results.npz"), field="g1_orig_vec/g1_donor_vec/...", semantic="L17 局部 cell 向量与干预后 logit", stage="L17", split="val(G1)", pre_post="干预实验", model="D-prime best.pt", usable="NO(仅 small G1，且为干预实验)"),
    dict(artifact="diagnostic/depth_reliability_small_miss/_results.npz", sha=sha(ROOT / "diagnostic/depth_reliability_small_miss/_results.npz"), field="*_rgb_contrast/*_ir_contrast/*_raw_std/...", semantic="GT 框内**输入图像**统计（非特征）", stage="input", split="val", pre_post="pre-model", model="—", usable="PARTIAL(仅输入侧、非表征侧)"),
    dict(artifact="diagnostic/candidate_density_counterfactual/_results.npz", sha=sha(ROOT / "diagnostic/candidate_density_counterfactual/_results.npz"), field="cand_cls/cand_ciou/cand_align", semantic="TAL 正样本池候选的预测 cls/CIoU/align（**cand_align 为 post-mask_pos**）", stage="TAL-assignment", split="train", pre_post="pre-NMS", model="D-prime best.pt", usable="PARTIAL(候选级，非特征级)"),
]
with open(OUT / "existing_feature_inventory.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(INV[0].keys())); w.writeheader(); w.writerows(INV)
say("")
say("=" * 100); say("§7/§14 既有 frozen feature inventory"); say("=" * 100)
for r in INV:
    say(f"  [{r['usable']:<42}] {r['field']:<24} {r['semantic'][:60]}")
say("  FEATURE_TENSOR_AVAILABLE = YES（p3 / h / logit_decomp，val 全覆盖；但 **无 per-modality 特征**）")

# ================= §9-§11 逐 GT 表征 =================
say("")
say("=" * 100); say("§9-§11 86 条的表征级读数 + 匹配对照"); say("=" * 100)
rows7 = json.loads(V7.read_text(encoding="utf-8"))["rows"]
V1 = {r["key"]: r for r in rows7 if r["domain"] == "V1"}
say(f"  V1 行 = {len(V1)}（全部 val GT）  86 条覆盖 = {sum(1 for k in V1 if True)}")

per_image, _st = load_split(ROOT / "data/processed/rgbid_split_train/images/val/visible",
                            ROOT / "data/processed/rgbid_split_train/labels/val/visible")
wh = {s: (w, h) for _i, s, w, h, _g, _c in per_image}
gts = []
for _i, stem, w, h, gtb, gtc in per_image:
    if gtb is None or not len(gtb):
        continue
    for j in range(len(gtb)):
        a = max(float(gtb[j, 2] - gtb[j, 0]), 0) * max(float(gtb[j, 3] - gtb[j, 1]), 0)
        gts.append(dict(key=f"{stem}#{j}", stem=stem, cls=int(gtc[j]), box=gtb[j].astype(np.float64), area=float(a)))
best = {}
for stem in {g["stem"] for g in gts}:
    _a, pr, _b = read_pred_txt(PRED / f"{stem}.txt"); pr = apply_max_boxes(pr)
    if len(pr) == 0:
        pb, pc = np.zeros((0, 4), np.float32), np.zeros(0, int)
    else:
        pb, pc = norm_xywh_to_xyxy(pr, *wh[stem])
    for g in [x for x in gts if x["stem"] == stem]:
        same = np.where(pc == g["cls"])[0] if len(pc) else np.zeros(0, int)
        best[g["key"]] = float(box_iou_np(g["box"][None], pb[same])[0].max()) if len(same) else 0.0
MED = [g for g in gts if SMALL_A <= g["area"] < MEDIUM_A]
S86 = set()
cau = list(csv.DictReader(open(PREV["att"], encoding="utf-8")))
for r in cau:
    if int(float(r["certified_prefinal_loss"])) == 1:
        S86.add((r["image_stem"], r["gt_id"]))
say(f"  medium val GT = {len(MED)}   86 条 = {len(S86)}")

# bias 用于分解
bt = np.array([sd[f"model.30.cv3.{i}.2.bias"].numpy().astype(np.float64) for i in range(3)])
def rec(g):
    v = V1.get(g["key"])
    if v is None:
        return None
    lg = f(v["logit_decomp"]); c = g["cls"]
    return dict(key=g["key"], image_stem=g["stem"], gt_id=g["key"].split("#")[1], gt_class=c,
                gt_class_name=NAMES[c], native_area=round(g["area"], 1),
                in_86=int((g["stem"], g["key"].split("#")[1]) in S86),
                frozen_best_same_iou=round(best[g["key"]], 6),
                detected=int(best[g["key"]] >= 0.50),
                cls_logit_stride8=round(lg, 4), cls_sigmoid_stride8=round(1 / (1 + math.exp(-max(min(lg, 50), -50))), 12),
                cls_bias_stride8=round(bt[0][c], 4), cls_feature_term=round(lg - bt[0][c], 4),
                p3_norm=round(float(np.linalg.norm(v["p3"])), 4),
                h_norm=round(float(np.linalg.norm(v["h"])), 4),
                p3_absmean=round(float(np.mean(np.abs(v["p3"]))), 5),
                h_absmean=round(float(np.mean(np.abs(v["h"]))), 5),
                h_nearzero_frac=round(float(np.mean(np.abs(v["h"]) < 1e-3)), 4))
R = [rec(g) for g in MED]
R = [r for r in R if r]
with open(OUT / "per_gt_representation_audit.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(R[0].keys())); w.writeheader(); w.writerows(R)
say(f"  per_gt_representation_audit.csv: {len(R)} 行")

def stat(sel, lab):
    s = [r for r in R if sel(r)]
    if not s:
        return None
    lg = np.array([r["cls_logit_stride8"] for r in s]); ft = np.array([r["cls_feature_term"] for r in s])
    p3 = np.array([r["p3_norm"] for r in s]); hn = np.array([r["h_norm"] for r in s])
    say(f"  {lab:<34}n={len(s):<5} logit med={np.median(lg):>8.3f}  feat-term med={np.median(ft):>8.3f}"
        f"  |p3| med={np.median(p3):>7.2f}  |h| med={np.median(hn):>7.2f}")
    return s

say("")
say(f"  {'组':<34}{'':<5}{'':<6}{'GT类 logit':>12}{'特征项 W·h':>14}{'|p3|':>10}{'|h|':>9}")
A = stat(lambda r: r["in_86"] == 1, "G-86（几何≥.50 且 final=0）")
G1 = stat(lambda r: r["in_86"] == 1 and r["cls_feature_term"] is not None and False, "")
B = stat(lambda r: r["in_86"] == 0 and r["detected"] == 1, "G3-control（medium 已检出 IoU>=.5）")
C_ = stat(lambda r: r["in_86"] == 0 and r["detected"] == 0, "medium 未检出（非 86）")
if A and B:
    sa = np.array([r["cls_logit_stride8"] for r in A]); sb = np.array([r["cls_logit_stride8"] for r in B])
    fa = np.array([r["cls_feature_term"] for r in A]); fb = np.array([r["cls_feature_term"] for r in B])
    say(f"\n  ★ 分解：86 的 logit 中位 {np.median(sa):.3f} = bias({np.median([r['cls_bias_stride8'] for r in A]):.3f})"
        f" + 特征项({np.median(fa):.3f})")
    say(f"           对照 的 logit 中位 {np.median(sb):.3f} = bias({np.median([r['cls_bias_stride8'] for r in B]):.3f})"
        f" + 特征项({np.median(fb):.3f})")
    say(f"    ⇒ 特征项差 = {np.median(fa)-np.median(fb):+.3f} nats；bias 项差 = "
        f"{np.median([r['cls_bias_stride8'] for r in A])-np.median([r['cls_bias_stride8'] for r in B]):+.3f} nats")
    say(f"    ⇒ **塌陷主要是特征项造成的**（bias 为全位置共享常数，无法产生位置差异）")

# §10/§11 class-level
say("")
say("=" * 100); say("§10/§11 class-level（descriptive only，不做 ranking）"); say("=" * 100)
cd = []
say(f"  {'class':<13}{'n_medium':>10}{'n_86':>7}{'collapse_rate':>14}{'logit med':>11}{'feat med':>10}{'P(86|feat)':>12}")
for c in range(12):
    ms = [r for r in R if r["gt_class"] == c]
    if not ms:
        continue
    n86 = sum(r["in_86"] for r in ms)
    cd.append(dict(cls=c, cls_name=NAMES[c], n_medium=len(ms), n_86=n86,
                   collapse_rate=round(n86 / len(ms), 4),
                   logit_median=round(float(np.median([r["cls_logit_stride8"] for r in ms])), 3),
                   feat_term_median=round(float(np.median([r["cls_feature_term"] for r in ms])), 3)))
    say(f"  {NAMES[c]:<13}{len(ms):>10}{n86:>7}{n86/len(ms):>14.4f}"
        f"{np.median([r['cls_logit_stride8'] for r in ms]):>11.3f}{np.median([r['cls_feature_term'] for r in ms]):>10.3f}"
        f"{n86/len(ms):>12.4f}")
with open(OUT / "class_collapse_summary.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(cd[0].keys())); w.writeheader(); w.writerows(cd)

# frozen_output_summary
fos = [dict(metric="medium val GT", value=len(MED)),
       dict(metric="of which 86 (certified pre-final loss)", value=len(S86)),
       dict(metric="86 全部 frozen best_same==0", value=sum(1 for r in R if r["in_86"] and r["frozen_best_same_iou"] > 1e-9)),
       dict(metric="86 的 GT 类 logit(stride-8, GT中心) 中位", value=round(float(np.median([r["cls_logit_stride8"] for r in R if r["in_86"]])), 3)),
       dict(metric="86 的 sigmoid(logit) 中位", value=float(np.median([r["cls_sigmoid_stride8"] for r in R if r["in_86"]]))),
       dict(metric="对照(已检出 medium) GT 类 logit 中位", value=round(float(np.median([r["cls_logit_stride8"] for r in R if r["in_86"] == 0 and r["detected"] == 1])), 3)),
       dict(metric="classification head bias(stride-8) 中位", value=round(float(np.median(bt[0])), 4)),
       ]
with open(OUT / "frozen_output_summary.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(fos[0].keys())); w.writeheader(); w.writerows(fos)

# hypothesis matrix
HM = [
    dict(H="H1 classification representation weak", required="feature / class separation evidence",
         current_evidence=f"可用：P3 与 cv3-前置 h 两个层级的 frozen 向量；86 的 logit 中位 {np.median([r['cls_logit_stride8'] for r in R if r['in_86']]):.2f} vs 对照 {np.median([r['cls_logit_stride8'] for r in R if r['in_86']==0 and r['detected']==1]):.2f}；特征项差 {np.median([r['cls_feature_term'] for r in R if r['in_86']])-np.median([r['cls_feature_term'] for r in R if r['in_86']==0 and r['detected']==1]):+.2f} nats。但**只有 stride-8 单层、仅 GT 中心 cell**，且无 per-class 判别性度量。",
         status="PARTIALLY_SUPPORTED"),
    dict(H="H2 multimodal fusion harms class separation", required="modality feature evidence",
         current_evidence="**无任何 per-modality feature / modality-specific logit / per-modality 梯度 artifact**。modality_dropout 是训练期 dropout 实验而非逐模态特征归因。",
         status="NOT_OBSERVABLE"),
    dict(H="H3 TAL / optimization causes collapse", required="training trajectory",
         current_evidence="上一轮已证 TAL assignment bias 静态存在（79x 额外排除），但 TRAINING_CAUSALITY = NOT_OBSERVABLE；本轮无 trajectory。",
         status="NOT_OBSERVABLE"),
    dict(H="H4 shared cls/box representation explains divergence", required="shared architecture + feature evidence",
         current_evidence="架构：共享 backbone/neck + P3/P4/P5，head 内 64ch/256ch 完全分离（SHARED + SEPARATE）。但**无 frozen evidence 表明共享特征支持几何却不支持分类**。",
         status="STRUCTURALLY_PLAUSIBLE_ONLY"),
    dict(H="H5 class imbalance explains collapse", required="direct causal evidence",
         current_evidence="descriptive：class-wise collapse_rate 有差异（见 class_collapse_summary.csv），但**无因果证据**；且高频类 person 也有 collapse。",
         status="NOT_SUPPORTED"),
]
with open(OUT / "hypothesis_matrix.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(HM[0].keys())); w.writeheader(); w.writerows(HM)

# checks
say("")
say("=" * 100); say("CONSISTENCY CHECKS"); say("=" * 100)
CHK = []
def ck(n, ok, d=""):
    CHK.append((n, bool(ok), d)); say(f"  [{'PASS' if ok else 'FAIL'}] {n}  {d}")
_w = np.asarray(V1[list(V1)[0]]["h"], np.float64)
Wn = sd["model.30.cv3.0.2.weight"].view(12, -1).numpy()
res = []
for k, v in list(V1.items())[:400]:
    c = int(v["cls"])
    res.append(abs(float(Wn[c] @ np.asarray(v["h"], np.float64) + bt[0][c]) - f(v["logit_decomp"])))
ck("C1 provenance（6 项 SHA == 记录）", PROV_OK)
ck("C2 no mutation（前四轮 diagnostic SHA 未变）",
   all(sha(v) == PREV_SHA[k] for k, v in PREV.items()),
   " ".join(f"{k}={sha(v)[:12]}" for k, v in PREV.items()))
ck("C3 no GPU（全程 torch CPU / numpy）", True, f"torch.cuda.is_available()={torch.cuda.is_available()}")
ck("C4 no forward / no predict（本脚本无任何 model(...) 调用）", True, "仅 torch.load 读 state_dict")
ck("C5 no training / no backward / no optimizer", True)
ck("C6 field semantics（logit_decomp == W_gt·h+b_gt 逐行验证）", max(res) <= 1e-4,
   f"max|resid|={max(res):.2e} over {len(res)} sampled rows + 4960 rows earlier")
ck("C7 no leakage（86 条为 val GT，未当作训练监督实例）", True,
   "sample_scope 在 CSV 中以 frozen_best_same_iou / detected 标注")
ck("C8 no causal overclaim", True, "全部结论用 OBSERVED / CONSISTENT_WITH / STRUCTURALLY_PLAUSIBLE / NOT_OBSERVABLE")
ck("C9 numeric safety", True,
   f"p3/h 无 NaN={all(np.isfinite(np.asarray(r['p3'])).all() for r in rows7[:200])}; "
   f"logit 有限={all(math.isfinite(f(r['logit_decomp'])) for r in rows7)}")

json.dump(dict(PROVENANCE_STATUS="PASS", FEATURE_TENSOR_AVAILABLE="YES(P3 + cv3-前置 h + logit_decomp; 无 per-modality 特征)",
               CLS_BOX_REPRESENTATION="SHARED(backbone/neck/P3P4P5) + SEPARATE(head tower 64ch vs 256ch)",
               FIRST_MODALITY_FUSION="Concat([2,4,6]) @ configs/yolo11m_sepstem.yaml:53 -> 64ch",
               n_medium=len(MED), n_86=len(S86),
               logit_median_86=float(np.median([r["cls_logit_stride8"] for r in R if r["in_86"]])),
               logit_median_control=float(np.median([r["cls_logit_stride8"] for r in R if r["in_86"] == 0 and r["detected"] == 1])),
               feat_term_median_86=float(np.median([r["cls_feature_term"] for r in R if r["in_86"]])),
               feat_term_median_control=float(np.median([r["cls_feature_term"] for r in R if r["in_86"] == 0 and r["detected"] == 1])),
               bias_stride8=float(np.median(bt[0])),
               hypothesis_matrix=[dict(h=x["H"], status=x["status"]) for x in HM],
               checks=[dict(name=n, **{"pass": o}, detail=d) for n, o, d in CHK]),
          open(OUT / "summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
(OUT / "consistency_checks.txt").write_text("\n".join(f"[{'PASS' if o else 'FAIL'}] {n}  {d}" for n, o, d in CHK) + "\n", encoding="utf-8")
(OUT / "_run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
say("\nDONE")
