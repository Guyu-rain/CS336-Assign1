# llm_from_scratch

最小训练/实验/生成入口都放在 `src/training/` 下，参数配置集中放在仓库根目录的 `configs/`，结果默认写到 `results/`。

## 1. 训练 BPE tokenizer

```bash
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.tokenizer.train_bpe \
  --config configs/train_bpe_tinystories.json
```

这会生成：
- `data/tokenizer/vocab.json`
- `data/tokenizer/merges.json`

## 2. 把文本转成训练用 `.npy`

```bash
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.prepare_data \
  --config configs/prepare_train.json
```

验证集同理：

```bash
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.prepare_data \
  --config configs/prepare_valid.json
```

## 3. 训练模型

训练指令现在只需要一行：

```bash
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.train_loop \
  --config configs/train_tiny_mps.json
```

说明：
- `device` 可用 `cpu`、`mps` 或 `cuda:0`
- 在 `mps` 上不要手动打开 TF32
- 如果是低资源设备，优先修改 `configs/train_tiny_mps.json`

## 4. 生成文本

```bash
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.generate \
  --config configs/generate_tiny_mps.json
```

## 5. 运行实验

学习率 sweep：

```bash
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.experiments \
  --config configs/experiment_lr_sweep.json
```

batch size sweep：

```bash
UV_NO_SYNC=1 UV_CACHE_DIR=.uv-cache uv run python -m src.training.experiments \
  --config configs/experiment_batch_size.json
```

## 6. `configs/` 目录

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

## 7. `results/` 目录结构

训练或生成后，结果会写成这种结构：

```text
results/
  tiny_mps/
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
```

其中：
- `metrics.jsonl` / `metrics.csv`：逐步 loss、学习率、wallclock time
- `loss_curve.svg`：最小可视化曲线
- `summary.json`：当前 run 的最好指标
- `samples/generated_text.txt`：生成文本
