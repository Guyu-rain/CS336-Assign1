# llm_from_scratch

最小训练、生成和实验入口都在 `src/training/` 下，配置集中放在仓库根目录的 `configs/`，默认输出目录是 `results/`。

当前这些配置已经恢复成面向 TinyStories 全量数据的版本：
- tokenizer 训练语料：`data/TinyStoriesV2-GPT4-train.txt`
- train token 数据：`data/tinystories_train_tokens.npy`
- valid token 数据：`data/tinystories_valid_tokens.npy`
- tokenizer 词表大小：`10000`
- 模型训练配置：2 层、`d_model=128`、`context_length=128`

## 0. 前置数据

先确保仓库根目录下存在这两个 TinyStories 文件：

```text
data/TinyStoriesV2-GPT4-train.txt
data/TinyStoriesV2-GPT4-valid.txt
```

如果你还没下载：

```bash
mkdir -p data
cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
cd ..
```

## 1. 从 BPE 训练开始的完整流程

1. 训练 TinyStories 的 BPE tokenizer

```bash
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.tokenizer.train_bpe \
  --config configs/train_bpe_tinystories.json
```

生成：
- `data/tokenizer/vocab.json`
- `data/tokenizer/merges.json`

2. 准备训练集 token

```bash
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.prepare_data \
  --config configs/prepare_train.json
```

生成：
- `data/tinystories_train_tokens.npy`

3. 准备验证集 token

```bash
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.prepare_data \
  --config configs/prepare_valid.json
```

生成：
- `data/tinystories_valid_tokens.npy`

4. 训练基础 TinyStories 模型

```bash
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.train_loop \
  --config configs/train_tiny_mps.json
```

默认生成：
- `results/tiny_mps/checkpoint.pt`
- `results/tiny_mps/config.json`
- `results/tiny_mps/metrics.jsonl`
- `results/tiny_mps/metrics.csv`
- `results/tiny_mps/loss_curve.svg`
- `results/tiny_mps/summary.json`

5. 基于训练好的 checkpoint 生成文本

```bash
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.generate \
  --config configs/generate_tiny_mps.json
```

默认生成：
- `results/tiny_mps_generate/config.json`
- `results/tiny_mps_generate/experiment_log.md`
- `results/tiny_mps_generate/samples/generated_text.json`
- `results/tiny_mps_generate/samples/generated_text.txt`

6. 跑 learning rate sweep

```bash
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.experiments \
  --config configs/experiment_lr_sweep.json
```

7. 跑 batch size sweep

```bash
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.experiments \
  --config configs/experiment_batch_size.json
```

## 2. 推荐执行顺序

如果你想从零开始完整跑完一次，按下面顺序执行即可：

```bash
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.tokenizer.train_bpe --config configs/train_bpe_tinystories.json
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.prepare_data --config configs/prepare_train.json
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.prepare_data --config configs/prepare_valid.json
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.train_loop --config configs/train_tiny_mps.json
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.generate --config configs/generate_tiny_mps.json
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.experiments --config configs/experiment_lr_sweep.json
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.experiments --config configs/experiment_batch_size.json
```

## 3. 设备说明

- 当前配置默认 `device` 是 `cpu`，这样在大多数机器上都能直接运行。
- 如果你在 Apple Silicon 上训练，可以把相关 config 里的 `device` 改成 `mps`。
- 如果你在 NVIDIA GPU 上训练，可以把 `device` 改成 `cuda:0`。
- `train_tiny_mps.json` 这个文件名保留了原命名，但里面现在是 TinyStories 全量训练配置，不再绑定 `mps`。

## 4. `configs/` 目录

```text
configs/
  train_bpe_tinystories.json
  prepare_train.json
  prepare_valid.json
  train_tiny_mps.json
  generate_tiny_mps.json
  experiment_lr_sweep.json
  experiment_batch_size.json
```

## 5. `results/` 目录结构

训练、生成或实验后，结果通常会写成类似结构：

```text
results/
  tiny_mps/
    checkpoint.pt
    config.json
    metrics.jsonl
    metrics.csv
    loss_curve.svg
    summary.json
    samples/
  tiny_mps_generate/
    config.json
    experiment_log.md
    samples/
      generated_text.json
      generated_text.txt
  tiny_lr_sweep_*/
  tiny_batch_sweep_*/
```

其中：
- `metrics.jsonl` / `metrics.csv`：逐步 loss、学习率、wallclock time
- `loss_curve.svg`：loss 曲线
- `summary.json`：当前 run 的汇总指标
- `samples/generated_text.txt`：生成文本样例
