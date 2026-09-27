# ASTRA-Tangent: Adaptive Subspace Updates for Tucker Fine-Tuning

ASTRA-Core **v0.2** adds **ASTRA-Tangent**, a research framework that decides when to update only a shared Tucker core and when to move selected mode subspaces on orthogonality-constrained manifolds.

**Status:** implemented and tested on synthetic tensors and a tiny offline BERT workflow. Real-task accuracy, GPU efficiency, and publication-level novelty remain to be established. The original ASTRA-Core / Mode / Hybrid implementations remain available as baselines.

Authors: Zhang Qingquan, Jiang Yurui. License: [BSD 3-Clause](LICENSE).

中文研究说明、文献对照、数学推导及投稿实验计划：**[RESEARCH_zh.md](docs/RESEARCH_zh.md)**。代码导读见 [CODE_MAP_zh.md](docs/CODE_MAP_zh.md)。

## Method

Compatible Transformer weights are stacked into family tensors, such as `[layer, head, head_dim, hidden]`. A pretrained Tucker decomposition supplies an initial core and orthonormal factors. The effective weight is

```text
W = W_pretrained + alpha * (Tucker(C0 + delta_C, U) - Tucker(C0, U0))
```

This preserves the original model at initialization and keeps its decomposition residual. The new optimizer:

1. Updates the core and classifier with AdamW.
2. Periodically probes horizontal factor gradients, removing directions already represented by the core.
3. Estimates each mode's first-order benefit using a damped core-Gram preconditioner.
4. Opens a small set of modes when their estimated benefit warrants moving the subspaces.
5. Uses polar retraction to keep updated factors orthonormal; inactive factors receive no updates.

Ranks stay fixed. The adaptation is in **which tangent blocks are updated**, not in rank growth. Standard geometric operations are not claimed as novel. The research hypothesis concerns task-driven allocation of mode-update effort. See the research note for exact equations, assumptions, and limitations.

## Install

Python 3.10+:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e .
```

Alternatively, install `requirements.txt`. Transformers 4.x is supported; 5.x is currently excluded. No new manifold package is required.

## Train

Provide a local Hugging Face sequence-classification model and a GLUE dataset saved with `datasets.save_to_disk()`:

```bash
python scripts/prepare_glue_dataset.py --task sst2 --output-dir ./local_datasets/glue_sst2

python -m astra_core \
  --model-path ./local_models/roberta-large \
  --dataset-path ./local_datasets/glue_sst2 \
  --glue-task sst2 --target-families q k v o \
  --attn-ranks 4 4 16 32 \
  --tuning-mode manifold --manifold-policy adaptive \
  --manifold-probe-interval 10 --manifold-warmup-steps 20 \
  --manifold-max-active-modes 2 --manifold-escape-threshold 0.15 \
  --learning-rate 1e-3 --num-train-epochs 3 \
  --run-name tangent_sst2
```

The multiline examples use Bash continuation; use PowerShell backticks or one line on Windows. `astra-core` and `python train_astra_core.py` are equivalent entry points.

Supported tasks: SST-2, MRPC, CoLA, QNLI, RTE, STS-B. Attention ranks describe `[layer, head, head_dim, hidden]`; FFN ranks describe `[stage, layer, output, input]` after orientation alignment.

The default tuning mode is now `manifold`. Select `--tuning-mode additive`, `multiplicative`, or `both` for the legacy baselines.

| Manifold policy | Behavior after core warmup |
|---|---|
| `adaptive` | Periodic gradient probes select up to the mode budget |
| `all` | Every nonsquare factor is updated |
| `core` | Factors remain fixed; same core optimizer as adaptive |
| `random` | Random modes with adaptive's selection count at the current state |

Additional controls include `--manifold-capture-fraction`, `--manifold-cost-power`, `--manifold-damping`, `--manifold-factor-lr-scale`, `--manifold-max-factor-step`, and `--gradient-accumulation-steps`. Use `--help` for defaults. Probe intervals and warmup are counted in optimizer steps.

## Ablations and artifacts

```bash
python -m astra_core \
  --model-path ./local_models/roberta-large \
  --dataset-path ./local_datasets/glue_sst2 \
  --sweep-config examples/sweep_tangent_sst2.json \
  --sweep-output-dir runs/tangent_sst2
```

The example includes controlled core/all/adaptive/random comparisons, three seeds, and selector ablations. It is a starting experiment configuration, not a completed benchmark.

Each run saves:

- `manifold_history.json`: probe scores, escape ratios, selected modes, captured energy, factor steps, and orthogonality errors.
- `experiment_config.json`, `experiment_summary.json`: configuration, validation results, training/evaluation wall time, and CUDA peak memory where available.
- `final/parameter_counts.json`: registered adapter parameters, factor parameters, horizontal degrees of freedom, and buffer bytes.
- `final/trainable_state.pt` and `final/adapter_basis.pt`: learned parameters plus the starting decomposition needed for portable adapter restoration.
- Existing CSV histories, plots, and sweep summaries.

Restore into the **same original CPU base model**, with matching classifier geometry:

```python
from transformers import AutoModelForSequenceClassification
from astra_core.checkpoint import load_adapter_checkpoint

model = AutoModelForSequenceClassification.from_pretrained('./local_models/roberta-large')
model = load_adapter_checkpoint(model, './astra_core_runs/sst2/tangent_sst2/final')
model.eval()  # move to the desired device after loading
```

Final adapter artifacts restore inference state; they are not Trainer resume checkpoints. The optimizer supports its own state serialization, including probe schedule and selected modes.

## Verification

```bash
python -m unittest discover -s tests -v
python scripts/smoke_train.py
```

The second command creates a random tiny BERT, a toy labeled dataset, runs all four policies through the real CLI, and reloads the exported adapters under `runs/offline_smoke/`. These data are not a real SST-2 benchmark.

Tests cover geometric directions, retraction, finite differences, selection, inactive-factor freezing, optimizer restoration, synthetic subspace learning, and tiny BERT training with gradient accumulation and attention/FFN tensor families.

## Research boundaries

The current runner targets single-process encoder classification on CPU or one GPU. AMP keeps manifold parameters in FP32. Multi-GPU/FSDP, quantization, decoder-only instruction tuning, dynamic rank changes, and RL post-training are not implemented or validated.

The learned Tucker tensor has mode-rank bounds `r_k`, but subtracting its initial anchor can produce update ranks up to `2*r_k`. All stored learnable factors count toward the parameter budget even when inactive on a particular step. Core AdamW means the full optimizer is not gauge invariant. Per-layer dense reconstruction and a dense anchor remain, so lower active update counts do not establish end-to-end speedups.

For related work including TensLoRA, Tucker dynamical low-rank training, StelLA, AdaLoRA, and ReFTA, and the experiments needed to distinguish this proposal from them, see [the research note](docs/RESEARCH_zh.md).
