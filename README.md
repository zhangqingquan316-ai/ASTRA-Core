# ASTRA-Core

Reference SFT code for the ASTRA blueprint's encoder-side Tucker family. In blueprint terms:

- `--tuning-mode additive` corresponds to `ASTRA-Core`
- `--tuning-mode multiplicative` corresponds to `ASTRA-Mode`
- `--tuning-mode both` corresponds to `ASTRA-Hybrid`

The current repository is a local-only research implementation for GLUE-style sequence classification. It is not yet the full ASTRA stack described in the blueprint. DPO, PPO, GRPO, ASTRA-G, and ASTRA-Muon are not implemented here.

Authors: Zhang Qingquan, Jiang Yurui

## 中文说明

这是一个预先公开的最小研究代码仓库版本。当前公开范围严格限定为：

- encoder-style GLUE classification 的 ASTRA-Core / ASTRA-Mode / ASTRA-Hybrid SFT 实现
- 不包含 DPO、PPO、GRPO、ASTRA-G、ASTRA-Muon
- 不包含模型权重、数据集、训练产物

中文代码导读见 [docs/CODE_MAP_zh.md](./docs/CODE_MAP_zh.md)。

## What This Repository Does

The main workflow:

- loads a local Hugging Face sequence-classification model,
- loads a locally saved GLUE dataset created with `datasets.save_to_disk()`,
- builds Tucker/HOOI decomposition caches for selected attention or FFN families,
- trains additive, multiplicative, or combined Tucker adapter parameters,
- saves trainable weights, metrics, histories, and parameter-count summaries.

The current code path is scoped to encoder-style classification models that expose `encoder.layer`, such as RoBERTa-like architectures.

## Repository Layout

```text
.
|-- README.md
|-- LICENSE
|-- pyproject.toml
|-- requirements.txt
|-- train_astra_core.py
|-- docs/
|   `-- CODE_MAP_zh.md
|-- examples/
|   |-- run_sst2.sh
|   `-- sweep_sst2.json
|-- scripts/
|   `-- prepare_glue_dataset.py
|-- tests/
|   `-- test_public_helpers.py
`-- astra_core/
    |-- __init__.py
    |-- __main__.py
    |-- artifacts.py
    |-- cli.py
    |-- constants.py
    |-- experiment.py
    |-- modeling.py
    |-- sweep.py
    |-- tasks.py
    `-- training_state.py
```

## Installation

Python `3.10+` is recommended.

```bash
git clone https://github.com/<your-name>/ASTRA-Core.git
cd ASTRA-Core
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

If you prefer not to install the package in editable mode:

```bash
pip install -r requirements.txt
```

## Prepare Local Inputs

### 1. Local model folder

`--model-path` must point to a local Hugging Face model directory.

### 2. Local GLUE dataset folder

The training code expects a dataset previously saved with `datasets.save_to_disk()`.

You can prepare it with:

```bash
python scripts/prepare_glue_dataset.py \
  --task sst2 \
  --output-dir ./local_datasets/glue_sst2
```

Supported GLUE tasks:

- `sst2`
- `mrpc`
- `cola`
- `qnli`
- `rte`
- `stsb`

## Quick Start

You can run the project in three equivalent ways.

### Option 1. Run the package module

```bash
python -m astra_core --help
```

### Option 2. Run the console script after `pip install -e .`

```bash
astra-core --help
```

### Option 3. Run the repository wrapper script

```bash
python train_astra_core.py --help
```

## Single-Run Example

```bash
python train_astra_core.py \
  --model-path ./local_models/roberta-large \
  --dataset-path ./local_datasets/glue_sst2 \
  --glue-task sst2 \
  --target-families q k v o \
  --attn-ranks 4 4 16 32 \
  --ffn-ranks 2 4 64 64 \
  --attn-alpha 1.0 \
  --ffn-alpha 1.0 \
  --tuning-mode additive \
  --multiplicative-num-bases 50 \
  --learning-rate 1e-3 \
  --per-device-train-batch-size 32 \
  --per-device-eval-batch-size 32 \
  --num-train-epochs 3 \
  --decomposition-cache-dir ./tucker_hooi_cache/roberta_large \
  --run-name astra_core_sst2
```

For Linux servers:

```bash
CUDA_VISIBLE_DEVICES=0 python -m astra_core \
  --model-path /data/models/roberta-large \
  --dataset-path /data/datasets/glue_sst2 \
  --glue-task sst2 \
  --target-families q k v o \
  --attn-ranks 4 4 16 32 \
  --ffn-ranks 2 4 64 64 \
  --tuning-mode both \
  --multiplicative-num-bases 50 \
  --learning-rate 1e-3 \
  --num-train-epochs 3 \
  --run-name astra_hybrid_sst2
```

To run in the background:

```bash
nohup CUDA_VISIBLE_DEVICES=0 python train_astra_core.py \
  --model-path /data/models/roberta-large \
  --dataset-path /data/datasets/glue_sst2 \
  --glue-task sst2 \
  --target-families q k v o \
  --attn-ranks 4 4 16 32 \
  --ffn-ranks 2 4 64 64 \
  --tuning-mode both \
  --multiplicative-num-bases 50 \
  --learning-rate 1e-3 \
  --num-train-epochs 3 \
  --run-name astra_hybrid_sst2 \
  > astra_hybrid_sst2.log 2>&1 &
```

## Sweep Example

An example sweep spec is provided in [examples/sweep_sst2.json](./examples/sweep_sst2.json).

```bash
python train_astra_core.py \
  --model-path ./local_models/roberta-large \
  --dataset-path ./local_datasets/glue_sst2 \
  --glue-task sst2 \
  --sweep-config ./examples/sweep_sst2.json \
  --sweep-output-dir ./runs/sst2_sweep
```

## Saved Outputs

For a single unnamed run, the code writes:

- `./astra_core_<task>_results/`
- `./astra_core_<task>_final/`

For named runs and sweeps, it writes:

- `./astra_core_runs/<task>/<run_name>/results/`
- `./astra_core_runs/<task>/<run_name>/final/`
- `./astra_core_sweeps/<task>_<timestamp>/`

Important saved artifacts include:

- `trainable_state.pt`
- `training_config.json`
- `parameter_counts.json`
- `experiment_config.json`
- `experiment_summary.json`
- `train_loss_history.csv`
- `eval_history.csv`

`parameter_counts.json` stores:

- `adapter_params`
- `classifier_trainable_params`
- `other_trainable_params`
- `total_trainable_params`
- `all_params`
- `trainable_ratio`

## Scope Notes

This repository is intentionally narrower than the full ASTRA blueprint. Before calling it a full public release, keep the scope statement explicit:

- implemented here: encoder-side GLUE classification SFT with ASTRA-Core, ASTRA-Mode, and ASTRA-Hybrid style parameterizations
- not implemented here: decoder-only instruction tuning, DPO, PPO, GRPO, ASTRA-G, ASTRA-Muon
- architecture assumption: models must expose `encoder.layer`

## Public Release Checklist

This repository is suitable for a minimal public release if you keep the framing narrow:

- publish it as the ASTRA-Core research-code repository, not as the full ASTRA project
- do not upload local models, datasets, checkpoints, or experiment outputs
- describe results only for experiments you can reproduce from this codebase
- keep the README scope statement intact so external readers know what is and is not implemented
