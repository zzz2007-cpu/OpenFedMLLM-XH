# Hateful Memes 数据集状态记录

更新时间：2026-05-16 17:34，环境：Windows conda `fl`。

## 数据来源

本次尝试使用 KaggleHub 下载：

```text
parthplc/facebook-hateful-meme-dataset
```

下载脚本：

```text
scripts/maintenance/download_hateful_memes_from_kaggle.py
```

KaggleHub 返回路径：

```text
C:\Users\31017\.cache\kagglehub\datasets\parthplc\facebook-hateful-meme-dataset\versions\1
```

## 当前校验结果

校验脚本：

```text
scripts/maintenance/refresh_hateful_memes_from_kaggle.py
```

校验报告：

```text
outputs/analysis/kaggle_hateful_memes_validation_report.json
```

日志：

```text
logs/hateful_memes_kaggle_refresh.log
```

后续按用户确认，改为采用 Kaggle 10k 版本替代项目原有 `hateful_memes/`，并基于 Kaggle `train.jsonl` 重新生成 federated split。

第一次按完整 seen/unseen split 校验失败，原因：

- Kaggle 目录实际包含 `train.jsonl`、`dev.jsonl`、`test.jsonl`。
- 未发现要求的 `dev_seen.jsonl`、`dev_unseen.jsonl`、`test_seen.jsonl`、`test_unseen.jsonl`。
- 图片数量为 10000，不是完整 seen/unseen split 所需的约 12540。
- `train.jsonl` 校验通过：8500 条，图片引用缺失数为 0，label 均为 0/1。

第二次按 Kaggle 10k 结构校验通过：

- `train.jsonl`：8500 条，图片引用缺失数 0，label 均为 0/1。
- `dev.jsonl`：500 条，图片引用缺失数 0，label 均为 0/1。
- `test.jsonl`：1000 条，图片引用缺失数 0；该 split 无 label。
- `img/`：10000 张图片。

## 当前项目数据状态

当前项目使用 Kaggle 10k 版本：

```text
hateful_memes/
  train.jsonl
  dev.jsonl
  test.jsonl
  dev_seen.jsonl   # dev.jsonl 的兼容副本
  test_seen.jsonl  # test.jsonl 的兼容副本，无 label
  img/
  federated/
  KAGGLE_SOURCE_MANIFEST.json
```

旧数据已从项目根目录替换掉，并保留备份：

```text
backup/hateful_memes_before_kaggle_10k_20260516_175858/hateful_memes/
```

## Federated Split 生成设置

项目已有生成脚本：

```text
scripts/build_hateful_memes_federated.py
```

如果未来拿到完整且校验成功的数据，推荐继续用以下设置生成：

- source split: `train`
- num clients: `10`
- seed: `42`
- stat settings: `iid`、`dir_1.0`、`dir_0.5`、`dir_0.1`
- modal settings: `aligned`、`missing_0.3`、`missing_0.4`、`missing_0.5`、`cross_3_7`、`cross_5_5`、`cross_7_3`、`hybrid_0.8`、`hybrid_0.7`、`hybrid_0.6`

当前已重新生成：

- 输出目录：`hateful_memes/federated/`
- setting 数量：40
- client 文件数量：400
- 每个 setting 总样本数：8500
- 校验报告：`outputs/analysis/hateful_memes_federated_validation_report.json`
- 校验结果：`success=true`

## Prompt Template

```text
Given the image and text, determine whether the meme is hateful.

Answer with only:
yes
or
no.

Text:
{text}
```

## Label Verbalizer

```text
0 -> no
1 -> yes
```

## 注意事项

- Kaggle 10k 版本没有 `dev_unseen.jsonl` 和 `test_unseen.jsonl`。
- `test.jsonl` / `test_seen.jsonl` 没有 label，不适合直接做有监督评估指标计算。
- MiniCPM Hateful Memes 训练默认使用 federated train split，不依赖 test label。
- 回滚旧数据：把 `backup/hateful_memes_before_kaggle_10k_20260516_175858/hateful_memes/` 复制或移动回项目根目录的 `hateful_memes/`。
