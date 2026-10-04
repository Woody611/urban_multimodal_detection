# PROVENANCE — Full-300 Training Trajectory Probe

```text
PROVENANCE_STATUS = PASS

BASELINE = D′ SepStem + IR-CLAHE（incumbent）
MODEL/LOSS/TAL/DATA/AUG/OPT/SCHED/SEED/BATCH/IMGSZ = UNCHANGED
PROBE_ONLY = YES
EARLY_STOP = NONE（§3/§4 要求跑满 300 epoch）
```

## §1 D′ provenance lock（逐项比对 PROBE30/args.yaml）

| 项 | expected | actual | |
|---|---|---|---|
| `epochs` | 300 | 300 | OK |
| `patience` | 0 | 0 | OK |
| `batch` | 8 | 8 | OK |
| `imgsz` | 1280 | 1280 | OK |
| `seed` | 42 | 42 | OK |
| `rect` | False | False | OK |
| `mosaic` | 1.0 | 1.0 | OK |
| `close_mosaic` | 10 | 10 | OK |
| `resume` | False | False | OK |
| `box` | 7.5 | 7.5 | OK |
| `cls` | 0.5 | 0.5 | OK |
| `dfl` | 1.5 | 1.5 | OK |
| `ir_encoding` | clahe | clahe | OK |

⇒ 冲突项：**0 个**（0 = 无冲突）

## 冻结 SHA256

| 文件 | SHA256 |
|---|---|
| model yaml | `9b14f2946073338414b4441784b6df87a868b982be32ac8d50b6d1c4a17f3dd8` |
| train config | `a4e329cfc3d220206448cdd3377c1b3f5126c50f13c907ada522d98725486e12` |
| train entry | `064a83e56e692203e9ae40a74703c992f831ebd5eed4f15e44136d70dca6ba9b` |
| loss | `0f092cf22a372f6d5c5e7197d7adde51cbb23c02ac4522978f42df5b23a6543b` |
| TAL | `aae7e8ac438f00cda88d7e151795b3f78270ec4ed56b883f413fd16233b7e527` |
| data base | `8bcf834930155bd58a99ccfc5e87113398f740aeb720b5827ff05ba45c5871d9` |
| augment | `53c75148fc168a5390b4f50523bcd9c1ba7379e5de6430f2875d8827953531c5` |
| D-prime best.pt | `1cae45f75693f54146e35c5fa076c0a6f78ca1cfa95595de74d959f40fae4fda` |
| medium_cells.json | `5d0ba324616bd48d8c7b9a01300eb867f4bc410ed26b78d247f62f2de5223c6b` |
| _probe.py (reused) | `a77fe65255f69be5fd9d6576c53d9b5ece007aadfec4cb473f528d8adf02ad30` |
| _probe300.py | `33c48de68c61f1a396b8efa2e00f4951fc9d62c09b047d4d6000ce3037e9412b` |
| _run_probe300.py | `ec2278d3709f4599c2ef0e26d63316980bc68371e69dd8931b4457e5640accd1` |
| _integrity_gate300.py | `b7680a4c5426616722e2eab3b9145236384657b4c798ba92c697d6bfd780a101` |
| _analyze300.py | `d7e441c5ccb010bbe2dd21cc629e8717818d891692e284ee12a30cb0fd2acd07` |
| _dryrun300.py | `27b4fb2184508750dc63b7591004b1f24f6e8f103ba28ca212fa976bb8ef5e00` |

## 观测点（§5）

```text
0-based : 0,1,2,5,10,20,30,50,75,100,125,150,175,200,225,250,275,289,299
csv     : 1,2,3,6,11,21,31,51,76,101,126,151,176,201,226,251,276,290,300
映射    : results.csv 写 self.epoch + 1（trainer.py:666）⇒ 0-based e ↔ csv e+1
§5 的 『290』『300』是 csv 口径 ⇒ 0-based 289 / 299（已避免 off-by-one）
close_mosaic=10 在 0-based 290 触发 ⇒ 观测点 275/289 在之前、299 在之后（§17）
```

## §2 状态恢复（HARD）

probe 期间临时 eval+head.train（为与既有 audit 特征口径可比）⇒ 每次观测后恢复：
train/eval 模式、BN buffers、全部参数（checksum）、RNG。G8 已实测（54 个 buffer 被改、还原后 0 个），
本轮另加 G10 在同一次操作里端到端验证四项恢复，PER-EPOCH 结果记入 trajectory_metadata.json。

## 与 PROBE30 的关系

- **未覆盖** PROBE30 的任何产物（新目录 diagnostic/full300_trajectory_probe/）。
- 复用 PROBE30 已验证的 P3TrajProbe/RawTAL 与角色标签（medium_cells.json，与既有 audit 同一文件）。
- **不早停**、观测点 9→19、新增 head_W/head_b 捕获与 CLS_OUT 恒等式（§8/§9C/§10）。
