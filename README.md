# ASTRA-Tangent: Adaptive Subspace Updates for Tucker Fine-Tuning

ASTRA-Core **v0.2** adds **ASTRA-Tangent**, a research framework that decides when to update only a shared Tucker core and when to move selected mode subspaces on orthogonality-constrained manifolds.

**Status:** implemented and tested on synthetic tensors and a tiny offline BERT workflow. Real-task accuracy, GPU efficiency, and publication-level novelty remain to be established. The original ASTRA-Core / Mode / Hybrid implementations remain available as baselines.

Authors: Zhang Qingquan, Jiang Yurui. License: [BSD 3-Clause](LICENSE).

The `jyr` branch adds direct Tucker increments, stable small-step geometry, and
a stateful coverage selector. CLI defaults and checkpoint layout handling are
documented in [the numerical stability note](docs/STABILITY_zh.md).

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
  --tuning-mode manifold --weight-layout delta \
  --manifold-policy stable --manifold-numerical-mode stable \
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
| `stable` | Smoothed energy shares, exact coverage subsets, hysteresis and confirmed switches (CLI default) |
| `adaptive` | Legacy density-greedy probes select up to the mode budget |
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

### Repeat the 34 rank settings across three tasks

`examples/sweep_ofat_cola_e15.json`, `examples/sweep_ofat_mrpc_e15.json`, and
`examples/sweep_ofat_sst2_e20.json` preserve all 34 attention/FFN rank pairs from
the existing SST-2 rank scan. They use adaptive manifold tuning on q/k/v/o and
FFN, learning rate 1e-3, train/eval batch size 32, and seed 42. CoLA and MRPC
run for 15 epochs; SST-2 runs for 20 epochs. Each run starts from the original
pretrained model. The runner evaluates CoLA with Matthews correlation and
MRPC/SST-2 with accuracy.

Prepare `local_datasets/glue_cola`, `glue_mrpc`, and `glue_sst2` with
`scripts/prepare_glue_dataset.py` if those saved datasets are missing. Activate
the existing training environment and select an available physical GPU (GPU 0
is only an example):

```bash
python scripts/run_ofat_glue.py --gpu 0 --dry-run
python scripts/run_ofat_glue.py --gpu 0
```

The launcher runs 102 experiments sequentially using the current Python
environment. It checks all dataset directories and the model config before
training, saves copies of the three sweep configurations, and writes task logs
and results under a new `astra_core_sweeps/ofat_glue_<timestamp>/` directory.

For the existing SSH server, Windows users can upload and start this OFAT
batch from the project directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_ofat_glue_remote.ps1
```

OpenSSH asks for the password in the local terminal. The starter discovers a
working training environment, selects an idle GPU, and starts a detached
launcher that prepares missing datasets and runs the batch. Existing dataset
directories are preserved. It prints the launcher PID, log, and result paths.
Use `-Gpu`, `-PythonPath`, `-ModelPath`, `-DatasetRoot`, or `-RemoteProject` for
explicit settings. The starter refuses occupied GPUs and detects an existing
batch launcher in this project.

### Learning-rate sweep with a shared warmup ratio

`examples/sweep_lr_warmup_cola_e15.json` and
`examples/sweep_lr_warmup_mrpc_e15.json` fix attention ranks to `[24,4,16,32]`
and FFN ranks to `[2,4,64,64]`. Each task runs six combinations of encoder
rates `[1e-3,2e-3]` and classifier rates `[1e-4,2e-4,5e-4]`, for 12 runs
in total. Each run uses 15 epochs, `warmup_ratio=0.06`, a linear schedule,
batch size 32, and seed 42. The two tasks run sequentially in one additional
training process.

In manifold mode, `--learning-rate` (alias `--encoder-learning-rate`) controls
the encoder adapter cores and factors. `--classifier-learning-rate` (alias
`--head-learning-rate`) sets the classifier's AdamW rate separately. When it
is omitted, the previous shared-rate behavior is retained. All groups use the
same warmup/decay multiplier; factor rates still include `factor_lr_scale`.
Saved configs, optimizer schedules, and sweep tables include both rates.

The CLI accepts `--warmup-ratio`, `--warmup-steps`, and `--lr-scheduler-type`.
Learning-rate warmup is separate from `--manifold-warmup-steps`, which delays
factor updates. The defaults retain zero LR warmup. Runtime logs and saved
metadata include the actual warmup and optimizer-step counts; training CSVs
also retain learning rates. MRPC records Accuracy, positive-class F1, and their
mean while continuing to select the best checkpoint by Accuracy.

On a server with enough spare GPU memory, run:

```bash
python scripts/run_lr_warmup_glue.py --gpu 0 \
  --model-path /absolute/path/to/local_models/roberta-large \
  --dataset-root /absolute/path/to/local_datasets \
  --decomposition-cache-dir /absolute/path/to/tucker_hooi_cache/roberta_large \
  --output-dir /absolute/path/to/new/lr_warmup_batch
```

The runner requires 12 GiB of free memory before starting each task and limits
its CUDA allocator to 48% of the GPU by default. It reuses the existing five
decomposition caches. Use `--dry-run` to inspect all 12 configurations.
For concurrent jobs, deploy this code as a separate snapshot so an existing
training process continues using its original implementation.
Use `--model-path`, `--dataset-root`, or `--output-dir` for custom paths.
An existing output directory is rejected. Training stops if a task fails;
check that task's log before starting another batch. On a remote server, run
the launcher inside a `tmux` session to keep it running after SSH disconnects.

## Remaining GLUE tasks at fixed rates

`examples/remaining_glue_e20.json` contains one run each for SST-2, QNLI,
RTE, and STS-B. All four use encoder LR `2e-3`, classifier LR `1e-4`,
warmup ratio `0.06`, 20 epochs, batch size 32, and seed 42. Attention ranks
remain `[24,4,16,32]` and FFN ranks remain `[2,4,64,64]`. STS-B uses one
regression output, floating labels, MSE loss, and Pearson correlation;
the other three use two classes and Accuracy.

```bash
python scripts/run_remaining_glue.py --gpu 0 \
  --model-path /absolute/path/to/local_models/roberta-large \
  --dataset-root /absolute/path/to/local_datasets \
  --decomposition-cache-dir /absolute/path/to/tucker_hooi_cache/roberta_large \
  --output-dir /absolute/path/to/new/remaining_glue_batch
```

Tasks run sequentially in the listed order. The launcher waits for at least
12 GiB of free memory before each task and uses the same 48% CUDA allocator
limit as the learning-rate batch. Use `--dry-run` to inspect the plan.
Each task has its own log, config, and result directory. `status.json`
records the current task and completed tasks. A used output directory is
rejected; an isolated code snapshot preserves concurrent experiments.

## RTE and STS-B learning-rate sweeps

`examples/sweep_lr_warmup_rte_e20.json` and
`examples/sweep_lr_warmup_stsb_e20.json` use the same six encoder/head pairs
and run names as CoLA/MRPC: encoder `[1e-3,2e-3]` crossed with classifier
`[1e-4,2e-4,5e-4]`. Both tasks run for 20 epochs with warmup ratio `0.06`
and the same attention/FFN ranks, batch size, and seed.

```bash
python scripts/run_lr_warmup_rte_stsb.py --gpu 0 \
  --model-path /absolute/path/to/local_models/roberta-large \
  --dataset-root /absolute/path/to/local_datasets \
  --decomposition-cache-dir /absolute/path/to/tucker_hooi_cache/roberta_large \
  --output-dir /absolute/path/to/existing/lr_warmup_batch
```

The launcher appends `rte/`, `stsb/`, their task configs, and logs to an
existing CoLA/MRPC batch. Its control files use the `rte_stsb_` prefix;
`rte_stsb_status.json` records progress. Existing RTE/STS-B artifacts cause
the launcher to refuse reuse. RTE uses Accuracy, and STS-B uses Pearson.
Both metrics are stored as `best_metric`, so the existing sorted result
table only needs `("rte", "Accuracy")` and `("stsb", "Pearson")` added to
its task loop. Use `--dry-run` to inspect the plan before submitting.

## RTE and STS-B OFAT extension

`examples/sweep_ofat_rte_e40.json` and `examples/sweep_ofat_stsb_e20.json`
reuse all 34 rank pairs, their order, and the `ofat_*` names from the original
SST-2 OFAT batch. RTE runs for 40 epochs and STS-B for 20. Encoder and head
share learning rate `1e-3`, with linear decay, no learning-rate warmup, and
attention/FFN alpha `1`, matching the original OFAT settings.

```bash
python scripts/run_ofat_rte_stsb.py --gpu 0 \
  --model-path /absolute/path/to/local_models/roberta-large \
  --dataset-root /absolute/path/to/local_datasets \
  --decomposition-cache-dir /absolute/path/to/isolated/roberta_large/cache \
  --output-dir /absolute/path/to/existing/ofat_glue_batch
```

The launcher adds `rte_config.json`, `stsb_config.json`, `rte/`, `stsb/`,
and task logs under the existing batch directory. Its status file is
`rte_stsb_ofat_status.json`. The server deployment uses a snapshot of the
original training code and seeds a separate decomposition cache from
existing cached ranks. A wrapper applies the CUDA allocator limit without
changing the training code. Tasks run sequentially, with a 12 GiB free-memory
check before each task. Use `--dry-run` to inspect all 68 planned runs.

## Estimate trainable parameter counts

`scripts/estimate_parameters.py` estimates RoBERTa adapter and classification
head counts without loading weights. In manifold mode, each family contributes
the product of its ranks for the core and `sum(dimension * rank)` for its
registered factors. Factors count as trainable even when the adaptive policy
leaves them inactive on a particular step. The head count includes both dense
layers; use `--num-labels 1` for STS-B regression.

```bash
python scripts/estimate_parameters.py --attn-ranks 24 4 16 32 \
  --ffn-ranks 2 4 64 64
mkdir -p runs
python scripts/estimate_parameters.py \
  --sweep-config examples/sweep_ofat_rte_e40.json \
  --output-csv runs/rte_parameter_estimates.csv
```

Use `--model-config` for another local RoBERTa configuration, or `--json` to
print structured counts. A sweep with `glue_task="stsb"` automatically uses
one output label unless explicitly overridden in its JSON entry.

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
