# Hateful Memes + MiniCPM-V 运行说明

## 1. 数据目录要求

默认数据根目录为：

```text
hateful_memes/
```

需要包含：

- `train.jsonl`
- `dev_seen.jsonl` 或其他 eval jsonl
- `img/`
- `federated/<stat_setting>/<modal_setting>/client_0.jsonl ... client_9.jsonl`
- `federated/<stat_setting>/<modal_setting>/meta.json`

本次接入不会移动或重写 `hateful_memes/` 下的任何原始数据。

## 2. Prompt Template

所有训练、验证、测试统一使用同一个 prompt：

```text
Given the image and text, determine whether the meme is hateful.

Answer with only:
yes
or
no.

Text:
{text}
```

## 3. Label Verbalizer

固定映射：

```text
0 -> no
1 -> yes
```

如果遇到非 `0/1` 标签，loader 会直接报错。

## 4. 可用联邦划分设置

当前实际检测到的 `stat_setting`：

- `iid`
- `dir_1.0`
- `dir_0.5`
- `dir_0.1`

当前实际检测到的 `modal_setting`：

- `aligned`
- `missing_0.3`
- `missing_0.4`
- `missing_0.5`
- `cross_3_7`
- `cross_5_5`
- `cross_7_3`
- `hybrid_0.8`
- `hybrid_0.7`
- `hybrid_0.6`

实际训练路径形如：

```text
hateful_memes/federated/dir_0.1/aligned/client_0.jsonl
```

## 5. Smoke Test

轻量数据链路检查：

```bash
python3 scripts/maintenance/test_hateful_memes_minicpm_pipeline.py \
  --data_root hateful_memes \
  --stat_setting iid \
  --modal_setting aligned \
  --client_id 0
```

Windows conda `fl` 环境推荐写法：

```bat
D:\Anaconda\condabin\conda.bat run -n fl python scripts\maintenance\test_hateful_memes_minicpm_pipeline.py --data_root hateful_memes --stat_setting iid --modal_setting aligned --client_id 0
```

该命令会检查：

- raw sample 是否可读
- image path 是否能拼接且存在
- prompt 是否按固定模板生成
- label 是否映射成 `yes/no`
- 指定 client split 是否能读取

如需进一步检查 MiniCPM preprocessing，可加：

```bash
python3 scripts/maintenance/test_hateful_memes_minicpm_pipeline.py \
  --data_root hateful_memes \
  --stat_setting iid \
  --modal_setting aligned \
  --client_id 0 \
  --check_minicpm_preprocess
```

这个模式会加载 MiniCPM-V，可能需要 GPU、模型缓存和 `bitsandbytes`。

## 6. 正式训练命令示例

推荐脚本入口：

```bash
stat_setting=dir_0.1 \
modal_setting=aligned \
rounds=20 \
client_sample_num=5 \
local_epochs=1 \
batch_size=1 \
lora_r=8 \
lora_alpha=16 \
lora_dropout=0.05 \
bash scripts/run_hateful_memes_minicpm.sh
```

也可以直接使用 registry 名称：

```bash
OPENFED_HATEFUL_MEMES_STAT=dir_0.1 \
OPENFED_HATEFUL_MEMES_MODAL=aligned \
OPENFED_NUM_ROUNDS=20 \
OPENFED_SAMPLE_CLIENTS=5 \
python3 mllmzoo/run_experiment.py --name minicpm_v_2_6_int4_hateful_memes_fedavg
```

Windows conda `fl` 环境示例：

```bat
set OPENFED_HATEFUL_MEMES_STAT=dir_0.1
set OPENFED_HATEFUL_MEMES_MODAL=aligned
set OPENFED_NUM_ROUNDS=20
set OPENFED_SAMPLE_CLIENTS=5
D:\Anaconda\condabin\conda.bat run -n fl python mllmzoo\run_experiment.py --name minicpm_v_2_6_int4_hateful_memes_fedavg
```

快速配置用于先确认完整 MiniCPM-V 训练链路能启动，默认 `num_rounds=1`、`sample_clients=2`、每个 client `max_steps=1`、eval 最多 32 条：

```bash
python3 mllmzoo/run_experiment.py --name minicpm_v_2_6_int4_hateful_memes_fedavg_quick
```

Windows conda `fl` 环境快速配置：

```bat
set OPENFED_HATEFUL_MEMES_STAT=dir_0.1
set OPENFED_HATEFUL_MEMES_MODAL=aligned
D:\Anaconda\condabin\conda.bat run -n fl python mllmzoo\run_experiment.py --name minicpm_v_2_6_int4_hateful_memes_fedavg_quick
```

默认输出目录：

```text
outputs/checkpoints/minicpmv_hateful_memes_<stat_setting>_<modal_setting>/
```

当前训练配置默认 `OPENFED_HATEFUL_MEMES_STRICT_IMAGE_PATH=0`，这样已有 federated split 中少量缺失图片会自动按 text-only 样本处理。若要严格检查图片路径并在缺失时立即报错，设置：

```bash
OPENFED_HATEFUL_MEMES_STRICT_IMAGE_PATH=1
```

## 7. 常见路径错误排查

- 找不到 client split：检查 `hateful_memes/federated/<stat_setting>/<modal_setting>/client_0.jsonl` 是否存在。
- 找不到图片：检查样本中的 `img` 字段是否类似 `img/xxxxx.png`，并确认文件存在于 `hateful_memes/img/`。
- label 报错：Hateful Memes loader 只接受 `0` 或 `1`。
- MiniCPM preprocessing 报错：确认模型依赖、显存、`bitsandbytes` 和 HF cache 是否正常。

## 8. 与 CrisisMMD 链路的对应关系

- CrisisMMD 训练入口：`mllmzoo/configs/minicpm/minicpmv-crisismmid-*.py`
- Hateful Memes 训练入口：`mllmzoo/configs/minicpm/minicpmv-hateful-memes-FedAvg.py`
- CrisisMMD task_type：默认 `classification`
- Hateful Memes task_type：`hateful_memes`
- CrisisMMD client 格式：通常为 `client_i.json`，内部已包含 `image` 和 `conversations`
- Hateful Memes client 格式：`client_i.jsonl`，内部是 raw `id/img/text/label`，由 loader 动态构造 `image` 和 `conversations`

本次没有改 CrisisMMD 的 loader、配置或数据路径。
