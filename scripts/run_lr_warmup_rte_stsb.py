"""Append six LR pairs each for RTE and STS-B to an existing GLUE batch."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_lr_warmup_glue import LEARNING_RATE_PAIRS, gpu_state, wait_for_memory, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("rte", "stsb")
PREFIX = "rte_stsb"


def build_plan(args):
    model = Path(args.model_path).expanduser().resolve()
    datasets = Path(args.dataset_root).expanduser().resolve()
    cache = Path(args.decomposition_cache_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    plans = []
    for task in TASKS:
        source = PROJECT_ROOT / "examples" / f"sweep_lr_warmup_{task}_e20.json"
        specs = json.loads(source.read_text(encoding="utf-8"))
        pairs = [(s["learning_rate"], s["classifier_learning_rate"]) for s in specs]
        if pairs != LEARNING_RATE_PAIRS:
            raise ValueError(f"Expected all six encoder/classifier LR pairs in {source}.")
        expected = dict(glue_task=task, num_train_epochs=20,
                        attn_ranks=[24, 4, 16, 32], ffn_ranks=[2, 4, 64, 64],
                        warmup_ratio=.06, warmup_steps=0, lr_scheduler_type="linear",
                        target_families=["q", "k", "v", "o", "ffn"],
                        tuning_mode="manifold", manifold_policy="adaptive",
                        per_device_train_batch_size=32, per_device_eval_batch_size=32,
                        gradient_accumulation_steps=1, max_length=128, weight_decay=.01, seed=42)
        for spec in specs:
            if any(spec.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Unexpected configuration in {source}.")
            encoder = format(spec["learning_rate"], ".0e").replace("e-0", "e-")
            head = format(spec["classifier_learning_rate"], ".0e").replace("e-0", "e-")
            name = f"enc_{encoder}_head_{head}_wr006_attn24-4-16-32_ffn2-4-64-64_s42"
            if spec.get("name") != name:
                raise ValueError(f"Run naming differs from CoLA/MRPC: {spec.get('name')}")
        config = output / f"{task}_config.json"
        command = [sys.executable, "-u", str(PROJECT_ROOT / "train_astra_core.py"),
                   "--model-path", str(model), "--dataset-path", str(datasets / f"glue_{task}"),
                   "--glue-task", task, "--decomposition-cache-dir", str(cache),
                   "--sweep-config", str(config), "--sweep-output-dir", str(output / task),
                   "--gpu-memory-fraction", str(args.gpu_memory_fraction)]
        plans.append(dict(task=task, runs=len(specs), epochs=20, specs=specs,
                          config_path=str(config), log_path=str(output / f"{task}.log"), command=command))
    return model, datasets, cache, output, plans


def check_output_available(output):
    # CoLA/MRPC use the unprefixed manifest/status/lock. Only reserve the new artifacts.
    reserved = [f"{PREFIX}_batch_manifest.json", f"{PREFIX}_status.json", f".{PREFIX}.launch.lock"]
    reserved += [name for task in TASKS for name in (task, f"{task}.log", f"{task}_config.json")]
    if any((output / name).exists() for name in reserved):
        raise FileExistsError(f"RTE/STS-B output already exists; refusing to overwrite: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--decomposition-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.48)
    parser.add_argument("--min-free-mib", type=int, default=12288)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpu < 0 or not 0 < args.gpu_memory_fraction <= 1 or args.min_free_mib <= 0:
        parser.error("Invalid GPU, memory fraction, or memory threshold.")
    model, datasets, cache, output, plans = build_plan(args)
    manifest = dict(gpu=args.gpu, gpu_memory_fraction=args.gpu_memory_fraction,
                    min_free_mib=args.min_free_mib, total_runs=sum(p["runs"] for p in plans),
                    tasks=plans, model_path=str(model), dataset_root=str(datasets),
                    decomposition_cache_dir=str(cache), code_snapshot=str(PROJECT_ROOT))
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        return
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"Missing model config: {model}")
    for task in TASKS:
        if not (datasets / f"glue_{task}" / "dataset_dict.json").is_file():
            raise FileNotFoundError(f"Missing saved dataset: {task}")
    for family in ("q", "k", "v", "o", "ffn"):
        ranks = "2-4-64-64" if family == "ffn" else "24-4-16-32"
        if not (cache / f"{family}_r{ranks}.pt").is_file():
            raise FileNotFoundError(f"Missing cached decomposition: {family}")
    check_output_available(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / f".{PREFIX}.launch.lock").open("x") as lock:
        lock.write(str(os.getpid()) + "\n")
    manifest["gpu_at_submission"] = gpu_state(args.gpu)
    write_json(output / f"{PREFIX}_batch_manifest.json", manifest)
    for plan in plans:
        write_json(Path(plan["config_path"]), plan["specs"])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    status_path = output / f"{PREFIX}_status.json"
    status = dict(state="submitted", launcher_pid=os.getpid(), completed_tasks=[])
    try:
        for plan in plans:
            status.update(state="waiting_for_memory", current_task=plan["task"], training_pid=None)
            write_json(status_path, status)
            wait_for_memory(args.gpu, args.min_free_mib)
            print(f"Starting {plan['task']}: six encoder/classifier LR pairs, 20 epochs, "
                  f"warmup_ratio=.06; log={plan['log_path']}", flush=True)
            with Path(plan["log_path"]).open("x", encoding="utf-8") as log:
                process = subprocess.Popen(plan["command"], cwd=Path.cwd(), env=environment,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                status.update(state="running", training_pid=process.pid)
                write_json(status_path, status)
                code = process.wait()
            if code:
                raise RuntimeError(f"{plan['task']} exited with code {code}; see {plan['log_path']}")
            status["completed_tasks"].append(plan["task"])
            status.update(training_pid=None)
            write_json(status_path, status)
            print(f"Completed {plan['task']}.", flush=True)
        status.update(state="complete", current_task=None, training_pid=None)
        write_json(status_path, status)
        print(f"All {manifest['total_runs']} runs completed. Results: {output}", flush=True)
    except Exception as error:
        status.update(state="failed", error=str(error))
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error
