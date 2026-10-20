"""Matched CoLA diagnostics; stopping preserves the original LR schedule.

Run from an isolated checkout. Output directories must be new. This does not
modify the source project, caches, or earlier experiment results.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import TrainerCallback
import astra_core.experiment as experiment
from astra_core.cli import parse_args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=["baseline", "numeric", "stable"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stop-epochs", type=int, default=20)
    p.add_argument("--schedule-epochs", type=int, default=40)
    p.add_argument("--head-lr", type=float, default=.005)
    options = p.parse_args()
    tag = f"{options.variant}_s{options.seed}_prefix{options.stop_epochs}"
    if (options.output / tag).exists():
        raise FileExistsError(f"Refusing to overwrite {options.output / tag}")
    options.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    class Monitor(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            if state.epoch >= options.stop_epochs:
                control.should_training_stop = True
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            with (options.output / f"{tag}_progress.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": state.global_step, "epoch": state.epoch,
                                         "metrics": metrics}) + "\n")
            return control

    original = experiment.Trainer
    def monitored_trainer(*args, **kwargs):
        kwargs["callbacks"] = list(kwargs.get("callbacks", [])) + [Monitor()]
        return original(*args, **kwargs)
    experiment.Trainer = monitored_trainer
    project = options.project
    baseline = options.variant == "baseline"
    sys.argv = [
        "stability_cola", "--model-path", str(project / "local_models/roberta-large"),
        "--dataset-path", str(project / "local_datasets/glue_cola"), "--glue-task", "cola",
        "--target-families", "q", "k", "v", "o", "ffn",
        "--attn-ranks", "24", "4", "16", "32", "--ffn-ranks", "2", "16", "64", "64",
        "--learning-rate", ".001", "--classifier-learning-rate", str(options.head_lr),
        "--warmup-ratio", ".06", "--num-train-epochs", str(options.schedule_epochs),
        "--weight-layout", "cpu_anchor" if baseline else "delta",
        "--manifold-policy", "stable" if options.variant == "stable" else "adaptive",
        "--manifold-numerical-mode", "legacy" if baseline else "stable",
        "--seed", str(options.seed), "--gpu-memory-fraction", ".45",
        "--decomposition-cache-dir", str(project / "tucker_hooi_cache/roberta_large"),
        "--run-name", tag,
    ]
    args = parse_args()
    args._resolved_decomposition_cache_dir = str(project / "tucker_hooi_cache/roberta_large")
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()}
    manifest["argv"] = sys.argv
    (options.output / f"{tag}_manifest.json").write_text(json.dumps(manifest, indent=2))
    experiment.run_single_experiment(args, 1, 1, sweep_root=options.output)


if __name__ == "__main__":
    main()
