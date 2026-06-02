# OpenFedMLLM

OpenFedMLLM 是一个面向多模态大模型的联邦微调与评估实验框架。项目以视觉语言任务为主要场景，支持将训练数据划分为多个客户端，在联邦学习设置下进行模型微调、参数聚合、评估与日志记录。

当前仓库主要围绕 MiniCPM-V、Qwen2-VL、CrisisMMD、Hateful Memes 和 VQAv2 等模型与数据集组织实验流程，同时保留了 Fling、Fling-LLM 和 FedMLLM 相关基础代码。

## 核心功能

- 多模态模型微调：支持 MiniCPM-V 和 Qwen2-VL 相关配置，主要通过 LoRA/PEFT 方式进行参数高效微调。
- 联邦训练算法：提供 FedAvg、FedProx、FedNova、SCAFFOLD 等联邦优化算法配置。
- 训练模式：支持 federated、centralized、local_only 等实验模式。
- 数据划分：提供 CrisisMMD、Hateful Memes、VQAv2 等数据集的客户端划分与整理脚本。
- 非 IID 实验：支持基于 Dirichlet 分布的客户端数据划分，并记录客户端样本量和标签分布。
- 评估与日志：记录准确率、F1、AUC、loss、通信量、轮次耗时、逐样本预测结果和解析失败情况。

## 目录结构

```text
.
├── fling/                         # Fling 联邦学习基础模块
├── fling_llm/                     # LLM 联邦训练相关代码
├── fling_mllm/                    # 多模态联邦训练、聚合与评估代码
│   ├── client/trainer/            # FedAvg/FedProx/FedNova/SCAFFOLD 训练器
│   ├── federated/                 # 参数聚合、联邦状态与 hook
│   ├── pipeline/                  # federated / centralized / local_only 流程
│   ├── tasks/                     # VQA、Hateful Memes、classification 任务适配
│   └── utils/                     # 模型构建、评估工具与预测解析
├── mllmzoo/                       # 多模态实验入口与配置管理
│   └── configs/                   # MiniCPM-V、Qwen2-VL 等实验配置
├── scripts/                       # 数据准备、划分、检查和维护脚本
├── partitions/                    # 部分联邦数据划分文件
├── hateful_memes/                 # Hateful Memes 数据与划分示例
├── docs/                          # 项目结构、数据集和运行说明文档
├── requirements.txt               # 基础依赖
└── requirements-minicpmv.txt      # MiniCPM-V 相关依赖
```

## 环境依赖

建议使用 Linux/WSL 和 NVIDIA GPU 环境运行完整训练流程。MiniCPM-V int4 训练依赖 CUDA、PyTorch、Transformers、PEFT 和 bitsandbytes。

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e .
pip install -r requirements-minicpmv.txt
```

如果需要匹配特定 CUDA 版本，请先安装对应版本的 PyTorch，再安装项目依赖。

## 数据准备

仓库不默认包含完整的 CrisisMMD、Hateful Memes、VQAv2 数据集和模型权重。运行训练前需要准备数据文件、图像文件和模型缓存。

MiniCPM-V CrisisMMD 主配置默认读取：

```text
data/crisis-mmd/minicpmv_data/
├── partition-alpha1.0-clt10/
│   ├── client_0.json
│   ├── client_1.json
│   └── ...
└── test.json
```

其中 `client_*.json` 是各客户端训练数据，`test.json` 是统一评估数据。

生成 CrisisMMD Dirichlet 非 IID 划分示例：

```bash
python scripts/generate_dirichlet_partitions.py \
  --alpha 0.5 \
  --num_clients 10 \
  --input_tsv crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv \
  --reference_dir data/crisis-mmd/minicpmv_data/partition-alpha1.0-clt10 \
  --output_dir data/crisis-mmd/minicpmv_data/partition-alpha0.5-clt10
```

脚本会输出划分文件和统计文件，用于查看客户端样本量、标签分布和数据异质性。

## 运行入口

MiniCPM-V CrisisMMD 提供统一 launcher，可通过参数选择算法、Dirichlet alpha、客户端数量和训练轮数。

Dry-run 检查：

```bash
python mllmzoo/configs/minicpm/run_minicpmv_crisismmid.py \
  --algorithm fedavg \
  --alpha 1.0 \
  --num-rounds 20 \
  --dry-run
```

启动训练：

```bash
python mllmzoo/configs/minicpm/run_minicpmv_crisismmid.py \
  --algorithm fedavg \
  --alpha 1.0 \
  --num-rounds 20
```

也可以直接运行具体配置文件：

```bash
python mllmzoo/configs/minicpm/minicpmv-crisismmid-FedAvg.py
python mllmzoo/configs/minicpm/minicpmv-crisismmid-FedProx.py
python mllmzoo/configs/minicpm/minicpmv-crisismmid-FedNova.py
python mllmzoo/configs/minicpm/minicpmv-crisismmid-Scaffold.py
```

Qwen2-VL 相关配置位于：

```bash
python mllmzoo/configs/qwen2vl/qwen2vl-crisismmid-FedAvg.py
python mllmzoo/configs/qwen2vl/qwen2vl-crisismmid-FedProx.py
python mllmzoo/configs/qwen2vl/qwen2vl-crisismmid-FedNova.py
python mllmzoo/configs/qwen2vl/qwen2vl-crisismmid-Scaffold.py
```

## 输出文件

MiniCPM-V CrisisMMD 默认将实验结果写入 `mllmzoo/output/`。常见输出包括：

- `eval_metrics_per_round.jsonl`：每轮或最终评估指标。
- `output_predictions.jsonl`：逐样本预测结果、标签、解析结果和错误类型。
- 控制台日志：训练轮次、loss、评估结果、通信量和耗时信息。
- rank log：多进程训练时可通过 `OPENFED_SPLIT_RANK_LOGS=1` 拆分日志。

示例：

```bash
OPENFED_SPLIT_RANK_LOGS=1 \
OPENFED_SPLIT_RANK_LOGS_TEE=1 \
python mllmzoo/configs/minicpm/run_minicpmv_crisismmid.py \
  --algorithm fedprox \
  --alpha 0.5 \
  --num-rounds 10
```

## 项目边界

- 本仓库用于研究和实验，不是生产级联邦学习系统。
- 当前没有实现差分隐私、安全聚合、TEE 或安全攻防评测。
- 完整训练依赖外部数据集、模型权重和 GPU 资源。
- 部分历史目录保留上游示例代码，当前多模态实验主线以 `fling_mllm/` 和 `mllmzoo/configs/` 为主。

## License

本仓库遵循 Apache 2.0 License。
