# OpenFedMLLM 项目结构（低风险迁移版）

## 1. 当前项目结构说明

本次是“低风险真实迁移”，遵循以下约束：

- 不修改核心训练/评估逻辑。
- 不删除任何原有文件。
- 不改训练入口和 import 结构。
- 通过“复制”而不是“移动”来集中外围目录。

项目当前仍以根目录结构为兼容主路径，训练入口保持不变，例如：

```bash
python3 mllmzoo/run_experiment.py --name minicpm_v_2_6_int4_fedavg
python3 mllmzoo/run_experiment.py --name qwen2_vl_2b_fedavg
```

## 2. 新增目录说明

本次新增目录：

- `outputs/`
- `outputs/logs/`
- `outputs/checkpoints/`
- `outputs/analysis/`
- `partitions/`
- `partitions/hateful_memes/`
- `partitions/crisismmd/`
- `partitions/vqav2/`
- `partitions/legacy_vqa_or_old/`
- `docs/`
- `scripts/maintenance/`

## 3. 已复制到新位置的目录

### 日志与输出

- `logging` -> `outputs/logs/logging`
- `loggs` -> `outputs/logs/loggs`

说明：旧目录保留，兼容旧脚本；新实验建议统一写入 `outputs/logs/`。

### 分析目录

- `analysis` -> `outputs/analysis/current`

说明：原 `analysis/` 保留，不改其中脚本逻辑。

### 联邦划分相关

- `partition-alpha0.1-clt10` -> `partitions/legacy_vqa_or_old/partition-alpha0.1-clt10`
- `partition-alpha0.5-clt10` -> `partitions/legacy_vqa_or_old/partition-alpha0.5-clt10`
- `partition-alpha1.0-clt10` -> `partitions/legacy_vqa_or_old/partition-alpha1.0-clt10`
- `partition_vqav2_supercat_dirichlet_clients10` -> `partitions/vqav2/supercat_dirichlet_clients10`
- `crisismmd_datasplit_all` -> `partitions/crisismmd/datasplit_all`

## 4. 旧目录仍保留

以下目录保持原位，不删除：

- `logging/`
- `loggs/`
- `analysis/`
- `partition-alpha0.1-clt10/`
- `partition-alpha0.5-clt10/`
- `partition-alpha1.0-clt10/`
- `partition_vqav2_supercat_dirichlet_clients10/`
- `crisismmd_datasplit_all/`

## 5. 暂不迁移（或部分迁移）目录及原因

- `hateful_memes/`：当前主数据目录，包含原始数据与图片，直接移动风险高。
- 仅复制了 `hateful_memes/federated` 下轻量级划分文件（`json/jsonl/txt/md/csv`）到 `partitions/hateful_memes/federated/`。
- 未复制 `hateful_memes/img/` 等图片重数据，避免高 I/O 和路径破坏风险。
- `test.json`：用途可能随实验变化，标记为 `UNKNOWN`，本次不迁移。

## 6. 当前推荐使用路径

- 新日志归档：`outputs/logs/`
- 新分析快照：`outputs/analysis/current/`
- 联邦划分整理视图：`partitions/`
- 兼容运行主路径：继续使用原根目录路径（尤其是已有配置中的硬编码路径）

## 7. 训练入口保护提醒

以下路径在代码/配置中有硬编码或强依赖，不建议在未做兼容层时直接修改：

- `./logging/...`
- `partition-alpha...`
- `partition_vqav2_supercat_dirichlet_clients10`
- `crisismmd_datasplit_all`
- `hateful_memes`

本次迁移未改动这些路径引用，保证原命令尽量可跑。

## 8. 回滚方式

本次迁移是“新增目录 + 复制数据”，未删除源目录，因此回滚非常直接：

1. 停用新目录：忽略 `outputs/`、`partitions/`、`docs/` 新增内容即可。
2. 如需彻底回退：删除本次新增的目标目录与文档（不影响源目录）。
3. 不需要恢复训练代码，因为本次未修改训练主流程。
