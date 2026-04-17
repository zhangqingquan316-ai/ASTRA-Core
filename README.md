# TensLoRA-XS

Research code for TensLoRA-XS style family-wise Tucker adapters with additive core updates and multiplicative mode transforms on local GLUE tasks.

This repository packages the `tenslora_xs` workflow into a GitHub-friendly structure so that other people can:

- understand the code layout,
- install the dependencies,
- prepare local GLUE datasets,
- run single experiments or sweeps,
- inspect saved adapter parameter counts and checkpoints.

## What This Repository Does

The main training workflow:

- loads a local Hugging Face sequence-classification model,
- loads a locally saved GLUE dataset created with `datasets.save_to_disk()`,
- builds Tucker/HOOI decomposition caches for selected attention or FFN families,
- trains either additive, multiplicative, or combined Tucker adapter parameters,
- saves trainable weights, metrics, histories, and parameter-count summaries.

This project is currently aimed at encoder-style classification models that expose `encoder.layer`, such as RoBERTa-like architectures.

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── train_tenslora_xs.py
├── archive/
│   └── legacy_monolith_backup.py
├── docs/
│   └── CODE_MAP_zh.md
├── examples/
│   ├── run_sst2.sh
│   └── sweep_sst2.json
├── scripts/
│   └── prepare_glue_dataset.py
└── tenslora_xs/
    ├── __init__.py
    ├── __main__.py
    ├── artifacts.py
    ├── cli.py
    ├── constants.py
    ├── experiment.py
    ├── modeling.py
    ├── sweep.py
    ├── tasks.py
    └── training_state.py
```

## Installation

Python `3.10+` is recommended.

```bash
git clone https://github.com/<your-name>/TensLoRA-XS.git
cd TensLoRA-XS
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

`--model-path` must point to a local Hugging Face model directory. For example, a RoBERTa checkpoint downloaded to disk.

### 2. Local GLUE dataset folder

The training code expects a dataset previously saved with `datasets.save_to_disk()`.

You can prepare it with:

```bash
python scripts/prepare_glue_dataset.py \
  --task sst2 \
  --output-dir ./local_datasets/glue_sst2
```

Supported GLUE tasks in this repository:

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
python -m tenslora_xs --help
```

### Option 2. Run the console script after `pip install -e .`

```bash
tenslora-xs --help
```

### Option 3. Run the repository wrapper script

```bash
python train_tenslora_xs.py --help
```

## Single-Run Example

```bash
python train_tenslora_xs.py \
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
  --run-name pm_additive_sst2
```

For Linux servers:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tenslora_xs \
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
  --run-name pm_both_sst2
```

To run in the background:

```bash
nohup CUDA_VISIBLE_DEVICES=0 python train_tenslora_xs.py \
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
  --run-name pm_both_sst2 \
  > pm_both_sst2.log 2>&1 &
```

## Sweep Example

An example sweep spec is provided in [examples/sweep_sst2.json](./examples/sweep_sst2.json).

```bash
python train_tenslora_xs.py \
  --model-path ./local_models/roberta-large \
  --dataset-path ./local_datasets/glue_sst2 \
  --glue-task sst2 \
  --sweep-config ./examples/sweep_sst2.json \
  --sweep-output-dir ./runs/sst2_sweep
```

## Saved Outputs

For a single run, the code writes outputs such as:

- `./tucker_lora_<task>_results/`
- `./tucker_lora_<task>_final/`

For a named run or sweep, it writes experiment folders such as:

- `./tucker_lora_runs/<task>/<run_name>/results/`
- `./tucker_lora_runs/<task>/<run_name>/final/`

Important saved artifacts include:

- `trainable_state.pt`
- `training_config.json`
- `parameter_counts.json`
- `experiment_config.json`
- `experiment_summary.json`
- `train_history.csv`
- `eval_history.csv`

`parameter_counts.json` stores:

- `adapter_params`
- `classifier_trainable_params`
- `other_trainable_params`
- `total_trainable_params`
- `all_params`
- `trainable_ratio`

## Project Assumptions And Limitations

- The current implementation expects encoder-based Hugging Face models with `encoder.layer`.
- The current entry point is for sequence classification rather than causal language modeling.
- The repository assumes local model folders instead of automatic remote downloads during training.
- The repository assumes local GLUE datasets already saved to disk.
- The code was organized to preserve the research logic, not to provide a fully generalized adapter library API.

## Additional Documentation

- Chinese code walkthrough: [docs/CODE_MAP_zh.md](./docs/CODE_MAP_zh.md)
- Original monolithic reference script: [archive/legacy_monolith_backup.py](./archive/legacy_monolith_backup.py)

## License

This repository is released under the MIT License. See [LICENSE](./LICENSE).
