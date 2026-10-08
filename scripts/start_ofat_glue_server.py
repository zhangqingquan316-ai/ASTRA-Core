"""Discover the server training environment and start one detached OFAT batch."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def training_processes():
    processes = []
    for directory in Path("/proc").glob("[0-9]*"):
        try:
            args = (directory / "cmdline").read_bytes().decode().split("\0")
            if not any(Path(arg).name in {"train_astra_core.py", "run_ofat_glue.py"}
                       or arg == "astra_core" for arg in args if arg):
                continue
            cwd = (directory / "cwd").resolve()
            env = dict(item.split("=", 1) for item in
                       (directory / "environ").read_bytes().decode().split("\0") if "=" in item)
            processes.append({"pid": int(directory.name), "args": args, "cwd": cwd, "env": env})
        except (OSError, UnicodeError):
            continue
    return processes


def select_gpu(requested):
    def query(arguments):
        return subprocess.run(
            ["nvidia-smi", *arguments], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        ).stdout

    busy = {line.split(",")[0].strip() for line in query([
        "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"
    ]).splitlines() if line.startswith("GPU-")}
    rows = query([
        "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ]).splitlines()
    for row in rows:
        index, uuid, used, utilization = [value.strip() for value in row.split(",")]
        if requested >= 0 and int(index) != requested:
            continue
        if uuid not in busy and used.isdigit() and int(used) < 512 and utilization == "0":
            return int(index)
    raise RuntimeError("No idle GPU available for this batch. Current GPU state:\n" + "\n".join(rows))


def select_python(explicit, processes, environment):
    candidates = [explicit] if explicit else []
    if not explicit:
        for process in processes:
            for key in ("VIRTUAL_ENV", "CONDA_PREFIX"):
                if process["env"].get(key):
                    candidates.append(str(Path(process["env"][key]) / "bin" / "python"))
            if process["args"][0].startswith("/"):
                candidates.append(process["args"][0])
        candidates.extend([str(PROJECT_ROOT / ".venv" / "bin" / "python"),
                           shutil.which("python"), sys.executable])
        for prefix in ("/opt/conda", "/root/miniconda3", "/root/anaconda3", "/root/miniforge3"):
            candidates.append(f"{prefix}/bin/python")
            candidates.extend(str(path) for path in Path(prefix).glob("envs/*/bin/python"))
    checked = set()
    errors = []
    probe = (
        "import torch, transformers, datasets, accelerate, tensorly; "
        "assert torch.cuda.is_available(), 'Selected GPU is unavailable to PyTorch'; "
        "print(torch.__version__)"
    )
    for candidate in candidates:
        if not candidate or candidate in checked or not Path(candidate).is_file():
            continue
        checked.add(candidate)
        result = subprocess.run([candidate, "-c", probe], env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, timeout=60)
        if result.returncode == 0:
            return candidate
        errors.append(f"{candidate}: " + (result.stderr.strip().splitlines() or ["probe failed"])[-1])
    raise RuntimeError("No usable training Python environment found. Pass --python-path.\n" + "\n".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=-1, help="Default: select an idle GPU.")
    parser.add_argument("--python-path")
    parser.add_argument("--model-path")
    parser.add_argument("--dataset-root")
    args = parser.parse_args()
    if args.gpu < -1:
        parser.error("--gpu must be -1 (automatic selection) or a nonnegative index.")
    processes = training_processes()
    for process in processes:
        if process["cwd"] == PROJECT_ROOT and any(
            Path(arg).name == "run_ofat_glue.py" for arg in process["args"] if arg
        ) and "--dry-run" not in process["args"]:
            print(f"A rank-sweep launcher is already running in this project: PID {process['pid']}.")
            return
    gpu = select_gpu(args.gpu)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    python_path = select_python(args.python_path, processes, environment)
    config_path = (PROJECT_ROOT / "astra_core_runs" / "sst2" /
                   "ofat_attn_baseline_r4-4-16-32" / "experiment_config.json")
    source = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    model_path = args.model_path or source.get("model_path") or "./local_models/roberta-large"
    dataset_root = args.dataset_root or "./local_datasets"
    if not args.dataset_root and source.get("dataset_path"):
        old_dataset = Path(source["dataset_path"])
        if old_dataset.name == "glue_sst2":
            dataset_root = str(old_dataset.parent)

    # Validate the exact 102-run plan without starting training or creating run outputs.
    batch_root = PROJECT_ROOT / "astra_core_sweeps"
    batch_name = f"ofat_glue_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output_dir = batch_root / batch_name
    command = [python_path, "-u", str(PROJECT_ROOT / "scripts" / "run_ofat_glue.py"),
               "--gpu", str(gpu), "--model-path", model_path,
               "--dataset-root", dataset_root, "--output-dir", str(output_dir),
               "--prepare-missing-datasets"]
    subprocess.run([*command, "--dry-run"], cwd=PROJECT_ROOT, env=environment, check=True)
    model_dir = Path(model_path).expanduser()
    if not model_dir.is_absolute():
        model_dir = PROJECT_ROOT / model_dir
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Model config missing: {model_dir / 'config.json'}")
    batch_root.mkdir(parents=True, exist_ok=True)
    log_path = batch_root / f"{batch_name}_launcher.log"
    with log_path.open("x") as log:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    (batch_root / f"{batch_name}_launcher.json").write_text(json.dumps({
        "pid": process.pid, "gpu": gpu, "python_path": python_path,
        "command": command, "log_path": str(log_path), "output_dir": str(output_dir),
    }, indent=2) + "\n", encoding="utf-8")
    time.sleep(2)
    returncode = process.poll()
    if returncode is not None and returncode != 0:
        raise RuntimeError(f"Launcher exited during startup; read {log_path}")
    status = "STARTED" if returncode is None else "COMPLETED"
    print(f"{status}: PID {process.pid}; GPU {gpu}; Python {python_path}", flush=True)
    print(f"Launcher log: {log_path}", flush=True)
    print(f"Results: {output_dir}", flush=True)
    print(f"View progress: tail -f {log_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        raise SystemExit(str(error)) from error
