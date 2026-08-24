# experiments/ — 实验工作目录

本目录按「实验」组织可复现材料（配置快照、结果数据、一次性脚本、图表）。

> **实验日志不写在这里**，统一记录在 [`docs/experiment_log.md`](../docs/experiment_log.md)。

## 目录职责

| 路径 | 放什么 | 是否入库 |
| :--- | :--- | :---: |
| `experiments/<实验名>/` | 该实验专属的配置快照、结果 JSON、一次性脚本、图表 | ✅ |
| `docs/experiment_log.md` | 实验记录（表格 + 详细记录 + 分析），唯一日志入口 | ✅ |
| `runs/<实验名>/` | 训练产物：权重、日志、tensorboard | ❌（已 gitignore） |
| `configs/` | 全局配置模板（dataset / model / train） | ✅ |

## 子目录规范

- 每个实验一个子目录，命名为 `<实验名>`（如 `baseline`、`fusion`），与 `docs/experiment_log.md` 的「实验编号 / 模型版本」对应；
- 子目录内可放：`config_snapshot.yaml`（该实验实际用的配置快照）、`metrics.json`（机器可读指标）、一次性脚本、可视化图表；
- 子目录**无需自带 README**；如需说明，在 `docs/experiment_log.md` 对应实验小节内补充，避免双份事实源。

## 新增实验流程

1. 在 `experiments/` 下新建 `<实验名>/` 子目录，放入配置快照等材料；
2. 训练产物自动写入 `runs/<实验名>/`；
3. 实验完成后，在 `docs/experiment_log.md` 的表格 + 详细记录 + 分析中登记结果。
