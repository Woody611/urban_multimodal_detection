# 模型权重（已随包提供）

本目录**已包含**训练好的权重，无需下载：

```
rematch/weights/best.pt    ← 本次提交所用权重
rematch/weights/last.pt    ← 第 300 epoch（对照）
```

## 校验

| 文件 | 字节数 | SHA-256 | 说明 |
|---|---:|---|---|
| `best.pt` | 40,643,241 | `5b99b6e5b20dca9a73e1183c5fda1cba2aba2d48d26366bff0704f530e911478` | **本次提交所用权重** |
| `last.pt` | 40,643,241 | `d2029dac8545a71e99e00d23dc2b4e32d733949b9a24c1c99e9f66720be2a455` | 第 300 epoch（对照） |

```bash
cd rematch && sha256sum weights/best.pt weights/last.pt
# 期望：
# 5b99b6e5b20dca9a73e1183c5fda1cba2aba2d48d26366bff0704f530e911478  weights/best.pt
# d2029dac8545a71e99e00d23dc2b4e32d733949b9a24c1c99e9f66720be2a455  weights/last.pt
```

## 另一份权重：预训练骨干 `yolo11m.pt`（也已随包提供）

**仅复现训练时需要**（推理/评测不需要）。它由 `configs/train_*.yaml` 的 `pretrained: "yolo11m.pt"`
引用，需放在**项目根目录**（与 `configs/` 同级），**不是本目录**：

```
rematch/
├── configs/
├── yolo11m.pt        ← 已随包提供（项目根目录，不是本目录）
└── weights/
    ├── best.pt       ← 本目录：训练好的权重
    └── last.pt
```

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `yolo11m.pt` | 40,684,120 | `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95` |

由 `configs/train_*.yaml` 的 `pretrained: "yolo11m.pt"` 引用。**已随包提供**；
Ultralytics 原版来源：`https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt`。

## 权重来源

`best.pt` 是 **D′ + IR-specific gamma/noise augmentation** 实验的 best checkpoint：

```
实验目录（原始）  runs/urban_multimodal_det_yolo11_rgbid_sepstem_clahe_iraug/
model config      configs/yolo11m_sepstem.yaml
train config      configs/train_rgbid_sepstem_clahe_iraug.yaml
seed              42
epochs            300（patience=0，无早停）
训练内 best       epoch 260（mAP50-95 0.56484）
```

自检（无需数据集）：

```bash
python - <<'PY'
import torch
ck = torch.load("rematch/weights/best.pt", map_location="cpu", weights_only=False)
ta = ck.get("train_args")
get = (lambda k: ta.get(k)) if isinstance(ta, dict) else (lambda k: getattr(ta, k, None))
print("ir_gamma            =", get("ir_gamma"))            # 期望 [0.75, 1.35]
print("ir_gamma_probability=", get("ir_gamma_probability"))# 期望 0.35
print("ir_noise_std        =", get("ir_noise_std"))        # 期望 0.015
print("ir_noise_probability=", get("ir_noise_probability"))# 期望 0.20
print("ir_encoding         =", get("ir_encoding"))         # 期望 clahe
print("channels            =", get("channels"))            # 期望 5
print("seed                =", get("seed"))                # 期望 42
sd = ck["model"].state_dict()
print("首层 conv shape      =", tuple(sd[list(sd)[0]].shape))  # 期望 (48, 3, 3, 3) = SepStem 的 RGB 分支
PY
```

四个 `ir_*` 键是本次实验的**唯一自变量**；`D′` 基线的同一位置是 `None`/默认值。
