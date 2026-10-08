"""Append the existing 34 OFAT rank settings for RTE and STS-B."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_lr_warmup_glue import gpu_state, wait_for_memory, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("rte", "stsb")
TASK_EPOCHS = {"rte": 40, "stsb": 20}


def check_output_available(output):
    reserved = ["rte_stsb_ofat_manifest.json", "rte_stsb_ofat_status.json", ".rte_stsb_ofat.launch.lock"]
    reserved += [name for task in TASKS for name in (task, f"{task}.log", f"{task}_config.json")]
    if any((output / name).exists() for name in reserved):
        raise FileExistsError(f"RTE/STS-B rank-search output already exists: {output}")


def build_plan(args):
    model = Path(args.model_path).expanduser().resolve()
    datasets = Path(args.dataset_root).expanduser().resolve()
    cache = Path(args.decomposition_cache_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    original = json.loads((output / "sst2_config.json").read_text(encoding="utf-8"))
    if len(original) != 34:
        raise ValueError("Expected the existing 34-run SST-2 OFAT configuration.")
    plans = []
    for task in TASKS:
        epochs = TASK_EPOCHS[task]
        source = PROJECT_ROOT / "examples" / f"sweep_ofat_{task}_e{epochs}.json"
        specs = json.loads(source.read_text(encoding="utf-8"))
        expected = [dict(spec, glue_task=task, num_train_epochs=epochs) for spec in original]
        if specs != expected:
            raise ValueError(f"Ranks, order, naming or settings differ from the existing batch: {source}")
        if any(spec["learning_rate"] != .001 for spec in specs):
            raise ValueError("Every OFAT run must use learning rate 1e-3.")
        config = output / f"{task}_config.json"
        command = [sys.executable, "-u", str(PROJECT_ROOT / "scripts/train_astra_core_limited.py"),
                   "--model-path", str(model), "--dataset-path", str(datasets / f"glue_{task}"),
                   "--glue-task", task, "--decomposition-cache-dir", str(cache),
                   "--sweep-config", str(config), "--sweep-output-dir", str(output / task)]
        plans.append(dict(task=task, runs=len(specs), epochs=epochs, specs=specs,
                          config_path=str(config), log_path=str(output / f"{task}.log"), command=command))
    return model, datasets, cache, output, plans


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
                    decomposition_cache_dir=str(cache), code_snapshot=str(PROJECT_ROOT),
                    encoder_learning_rate=.001, classifier_learning_rate=.001,
                    lr_scheduler_type="linear", warmup_ratio=0., attn_alpha=1., ffn_alpha=1.)
    if args.dry_run:
        print(json.dumps(manifest, indent=2), flush=True)
        return
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"Missing model config: {model}")
    for task in TASKS:
        if not (datasets / f"glue_{task}" / "dataset_dict.json").is_file():
            raise FileNotFoundError(f"Missing saved dataset: {task}")
    if not cache.is_dir():
        raise FileNotFoundError(f"Missing isolated decomposition cache: {cache}")
    check_output_available(output)
    with (output / ".rte_stsb_ofat.launch.lock").open("x") as lock:
        lock.write(str(os.getpid()) + "\n")
    manifest["gpu_at_submission"] = gpu_state(args.gpu)
    write_json(output / "rte_stsb_ofat_manifest.json", manifest)
    for plan in plans:
        write_json(Path(plan["config_path"]), plan["specs"])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["ASTRA_GPU_MEMORY_FRACTION"] = str(args.gpu_memory_fraction)
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    status_path = output / "rte_stsb_ofat_status.json"
    status = dict(state="submitted", launcher_pid=os.getpid(), completed_tasks=[])
    try:
        for plan in plans:
            status.update(state="waiting_for_memory", current_task=plan["task"], training_pid=None)
            write_json(status_path, status)
            wait_for_memory(args.gpu, args.min_free_mib)
            print(f"Starting {plan['task']}: 34 OFAT rank settings, LR=1e-3, {plan['epochs']} epochs; "
                  f"log={plan['log_path']}", flush=True)
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
        print(f"All 68 OFAT runs completed. Results: {output}", flush=True)
    except Exception as error:
        status.update(state="failed", error=str(error))
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error
