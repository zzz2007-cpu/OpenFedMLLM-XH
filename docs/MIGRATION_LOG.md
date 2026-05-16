# OpenFedMLLM 低风险迁移日志

## 1. 迁移时间

- 执行时间：2026-05-16 16:31:36 +0800
- 迁移类型：真实迁移（非 dry-run）
- 迁移原则：只复制、不删除、不改训练逻辑

## 2. 本次迁移动作列表

1. 创建新目录结构：`outputs/`、`partitions/`、`docs/`、`scripts/maintenance/` 及其子目录。
2. 复制日志目录到 `outputs/logs/`（保留原目录）。
3. 复制 `analysis/` 到 `outputs/analysis/current/`（保留原目录）。
4. 复制指定 partition 目录到 `partitions/` 下对应位置（保留原目录）。
5. 对 `hateful_memes/` 执行“轻量复制策略”：仅复制 federated split 元数据与 JSON/JSONL 等轻量文件，不复制图片数据。
6. 新增可复现迁移脚本：`scripts/maintenance/migrate_project_light.py`。

## 3. 源路径 -> 目标路径映射

- `logging` -> `outputs/logs/logging`
- `loggs` -> `outputs/logs/loggs`
- `analysis` -> `outputs/analysis/current`
- `partition-alpha0.1-clt10` -> `partitions/legacy_vqa_or_old/partition-alpha0.1-clt10`
- `partition-alpha0.5-clt10` -> `partitions/legacy_vqa_or_old/partition-alpha0.5-clt10`
- `partition-alpha1.0-clt10` -> `partitions/legacy_vqa_or_old/partition-alpha1.0-clt10`
- `partition_vqav2_supercat_dirichlet_clients10` -> `partitions/vqav2/supercat_dirichlet_clients10`
- `crisismmd_datasplit_all` -> `partitions/crisismmd/datasplit_all`
- `hateful_memes/federated`（轻量文件）-> `partitions/hateful_memes/federated`

## 4. 跳过项

- `hateful_memes/img/` 等大体积图片数据：跳过（防止重复制和路径风险）。
- `hateful_memes/` 根目录迁移：跳过（当前训练/数据主路径）。
- `test.json`：`UNKNOWN`，本次不迁移。
- `logs crisismmid-classfication-minicpm-0.5`（带空格目录名）：在当前环境未检测到独立同名目录，未执行该项复制。

## 5. 风险项

- 代码中存在大量硬编码路径（`logging`、`partition-alpha*`、`partition_vqav2...`、`crisismmd_datasplit_all`、`hateful_memes`）。
- `mllmzoo`、`zoo`、`FedMLLM` 多处配置使用固定相对路径或绝对路径。
- 若直接改写这些路径，可能破坏现有实验复现链路。

## 6. 后续建议

1. 先在配置层引入“可覆盖路径变量”（env/config），再逐步替换硬编码。
2. 优先改“新实验模板”，不要直接批量改历史配置。
3. 每次只迁移一个数据族（如先 CrisisMMD，再 VQAv2），并配套 smoke test。
4. 等路径兼容完成后，再考虑把旧目录标记为只读归档。
