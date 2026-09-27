# Validation record — 2026-09-28

Environment: Windows, Python 3.12.7, PyTorch 2.7.0+cpu, Transformers 4.57.6,
TensorLy 0.9.0, Datasets 3.6.0, Accelerate 1.15.0. Dependencies were installed
in the repository's ignored `.venv`; the system environment was not replaced.

Executed:

```text
python -m unittest discover -s tests -v
16 tests passed

python scripts/smoke_train.py
Four-policy CLI sweep and adapter reload passed

python -m compileall -q astra_core tests scripts train_astra_core.py
Passed

git diff --check
Passed
```

Coverage includes horizontal orthogonality, polar retraction, orthogonal gauge
equivariance of the geometric direction (not the entire AdamW optimizer),
finite-difference directional decrease, the ideal tangent energy decomposition,
rank-deficient numerical inputs, selector thresholds/budgets, factor freezing,
exact optimizer-state continuation, nonfinite-gradient skipping, CPU BF16
autocast initialization, and initial-model preservation for all legacy modes.

The synthetic learning test deliberately places a target outside the initial
mode subspace. Under its fixed seed and settings, adaptive and all-mode updates
both finish below one tenth of the core-only squared reconstruction error.
This is a mechanism regression test, not evidence of language-task quality.

A randomly initialized two-layer BERT test covers Q/K/V/O and FFN tensorization,
gradient accumulation, Trainer scheduling, and exact logits after exporting
and reloading an adapter into the original base model.

The CLI smoke run creates eight toy training examples and four toy validation
examples, executes adaptive/core/all/random, exercises decomposition caching,
best-state restoration, JSON/CSV/plot export, and checkpoint reload. Generated
models/data/results live under ignored `runs/offline_smoke/`. Their validation
accuracy must not be reported as a GLUE result.

No GPU benchmark or pretrained-model GLUE experiment was run. CUDA AMP,
multi-device execution, real-task improvement, wall-clock savings, and
publication claims have not been established by these checks.
