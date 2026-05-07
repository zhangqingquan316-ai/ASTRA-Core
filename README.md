# ASTRA-Core: Spectral Tensor Reparameterization for Efficient Adaptation

Hi!

This repository is the public research-code repository for **ASTRA-Core**, the first implemented stage of our broader **ASTRA** project, developed around the blueprint **"Parameters are Tensor Fields: Multilinear Geometry for Efficient LLM Adaptation and Post-Training."**

**ASTRA** stands for **A**daptive **S**pectral **T**ensor **R**eparameterized **A**lignment. The central idea is to stop treating Transformer parameters as isolated matrices and instead organize compatible weights into **family-wise tensor fields**, where structure across layers, projection types, and hidden dimensions can be modeled explicitly.

The current release focuses on the encoder-side supervised fine-tuning setting: family-wise Tucker/HOSVD decompositions, additive core updates, multiplicative mode transforms, and their hybrid combination for local GLUE classification tasks.

Authors: Zhang Qingquan, Jiang Yurui

-----

## 中文说明

这是 ASTRA 项目的第一阶段公开代码仓库，当前只公开已经实现并且边界清晰的部分：

- encoder-style GLUE classification 的 SFT 代码
- `ASTRA-Core`、`ASTRA-Mode`、`ASTRA-Hybrid` 三种参数化方式
- 不包含 DPO、PPO、GRPO、ASTRA-G、ASTRA-Muon
- 不包含模型权重、数据集和训练产物

中文代码导读见 [docs/CODE_MAP_zh.md](./docs/CODE_MAP_zh.md)。

-----

## What is ASTRA-Core?

Parameter-efficient fine-tuning methods such as LoRA usually adapt each projection matrix independently. That is effective, but it ignores the possibility that updates across layers and module families may share a higher-order multilinear structure.

**ASTRA-Core** starts from the observation that pretrained Transformer weights can be grouped into semantically meaningful **families** and stacked into high-order tensors. For example, compatible attention or FFN matrices can be organized along modes such as layer, module type, output dimension, and input dimension. A Tucker/HOSVD decomposition then provides a spectral multilinear subspace induced by the pretrained model.

Within that view, the current repository implements three related adaptation variants:

- **ASTRA-Core**: train only an additive Tucker-core update while keeping the mode factors fixed
- **ASTRA-Mode**: train multiplicative mode transforms over the shared Tucker core
- **ASTRA-Hybrid**: combine additive core updates with multiplicative mode transforms

This is intentionally narrower than the full ASTRA blueprint. The broader project also studies gradients, Hessians, optimizer-space projections, and post-training updates, but those components are not part of this repository yet.

-----

## Key Features

* **Family-Wise Tensorization**: Groups compatible attention or FFN weights into shared tensor families instead of treating every layer independently.
* **Spectral Reparameterization**: Builds Tucker/HOSVD decompositions from pretrained weights and performs adaptation inside the induced multilinear subspace.
* **Multiple Adaptation Variants**: Supports additive (`ASTRA-Core`), multiplicative (`ASTRA-Mode`), and hybrid (`ASTRA-Hybrid`) training modes in one codebase.
* **Local-First Research Workflow**: Uses local Hugging Face model folders and locally saved GLUE datasets, which keeps experiments reproducible and easy to audit.
* **Explicit Scope Boundaries**: Publishes the implemented encoder-side SFT part of ASTRA without over-claiming unsupported DPO, PPO, GRPO, or optimizer-level extensions.
* **Experiment Bookkeeping**: Saves trainable states, parameter-count summaries, histories, sweep outputs, and decomposition caches for later analysis.

-----

## Getting Started

### Installation

Clone the repository and install it as a package:

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

### Prepare Local Inputs

The current repository expects two local inputs:

1. **A local Hugging Face model folder**

   `--model-path` must point to a local sequence-classification model directory.

2. **A local GLUE dataset saved with `datasets.save_to_disk()`**

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

### Usage

You can run the project in three equivalent ways:

```bash
python -m astra_core --help
```

```bash
astra-core --help
```

```bash
python train_astra_core.py --help
```

To run a single experiment, for example:

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

To run a sweep:

```bash
python train_astra_core.py \
  --model-path ./local_models/roberta-large \
  --dataset-path ./local_datasets/glue_sst2 \
  --glue-task sst2 \
  --sweep-config ./examples/sweep_sst2.json \
  --sweep-output-dir ./runs/sst2_sweep
```

For more details on code organization and hyperparameters, please refer to the source code and the Chinese walkthrough in [docs/CODE_MAP_zh.md](./docs/CODE_MAP_zh.md).

-----

## Current Experimental Scope

The broader ASTRA blueprint is motivated by four research hypotheses:

- pretrained parameter families exhibit multilinear low-dimensional structure
- fine-tuning updates align with pretrained multilinear subspaces
- effective gradient and curvature directions can be projected into family-wise tensor subspaces
- similar geometry may extend from SFT to post-training regimes such as DPO, PPO, and GRPO

This public repository, however, intentionally publishes only the part that is currently implemented and reproducible:

- encoder-side GLUE classification experiments
- family-wise Tucker/HOSVD decompositions over attention and FFN families
- additive, multiplicative, and hybrid ASTRA adaptation variants
- parameter counting, decomposition caching, history logging, and sweep summaries

The following components belong to the broader ASTRA roadmap and are **not** included here:

- decoder-only instruction tuning
- DPO, PPO, and GRPO post-training
- ASTRA-G optimizer-level gradient projection
- ASTRA-Muon momentum orthogonalization
- the larger gradient/Hessian analysis toolkit described in the blueprint

This release should therefore be read as the **ASTRA-Core research-code baseline**, not as the complete ASTRA system.
