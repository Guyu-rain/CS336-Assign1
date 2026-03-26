# cs336_basics

最小训练/实验/生成入口都放在 `cs336_basics/training/` 下，结果默认写到仓库根目录的 `results/`。

## 1. 训练 tokenizer

```bash
uv run python -m cs336_basics.tokenizer.train_bpe \
  --input_path data/TinyStoriesV2-GPT4-train.txt \
  --vocab_size 10000 \
  --special_tokens "<|endoftext|>"
```

如果你已经用自己的脚本产出了 `vocab.json` 和 `merges.json`，后面直接复用即可。

## 2. 把文本转成训练用 `.npy`

```bash
uv run python -m cs336_basics.training.prepare_data \
  --input-path data/TinyStoriesV2-GPT4-train.txt \
  --output-path data/tinystories_train_tokens.npy \
  --vocab-path path/to/vocab.json \
  --merges-path path/to/merges.json \
  --special-token "<|endoftext|>"
```

验证集同理：

```bash
uv run python -m cs336_basics.training.prepare_data \
  --input-path data/TinyStoriesV2-GPT4-valid.txt \
  --output-path data/tinystories_valid_tokens.npy \
  --vocab-path path/to/vocab.json \
  --merges-path path/to/merges.json \
  --special-token "<|endoftext|>"
```

## 3. 训练模型

Apple Silicon / CPU 最小示例：

```bash
uv run python -m cs336_basics.training.train_loop \
  --train-data-path data/tinystories_train_tokens.npy \
  --val-data-path data/tinystories_valid_tokens.npy \
  --checkpoint-path results/tiny_mps/checkpoint.pt \
  --results-dir results \
  --run-name tiny_mps \
  --device mps \
  --batch-size 8 \
  --context-length 128 \
  --num-iterations 500 \
  --d-model 128 \
  --num-layers 2 \
  --num-heads 4 \
  --d-ff 384 \
  --vocab-size 10000 \
  --lr 3e-4 \
  --min-lr 3e-5 \
  --warmup-iters 50 \
  --cosine-cycle-iters 500 \
  --eval-every 50 \
  --eval-iters 10 \
  --log-every 10 \
  --checkpoint-every 100
```

说明：
- `device` 可用 `cpu`、`mps` 或 `cuda:0`
- 在 `mps` 上不要手动打开 TF32
- 如果是低资源设备，优先减小 `batch-size`、`context-length`、`num-iterations`、`d-model`

## 4. 生成文本

```bash
uv run python -m cs336_basics.training.generate \
  --checkpoint-path results/tiny_mps/checkpoint.pt \
  --vocab-path path/to/vocab.json \
  --merges-path path/to/merges.json \
  --special-token "<|endoftext|>" \
  --prompt "Once upon a time" \
  --max-new-tokens 256 \
  --temperature 0.8 \
  --top-p 0.9 \
  --device mps \
  --results-dir results \
  --run-name tiny_mps_generate \
  --d-model 128 \
  --num-layers 2 \
  --num-heads 4 \
  --d-ff 384 \
  --vocab-size 10000 \
  --context-length 128
```

## 5. `results/` 目录结构

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
