"""Summarize completed runs at a common epoch horizon (validation, not test).

Use history.json, not the final evaluation of the restored best checkpoint.
This keeps late-training collapse visible instead of hiding it behind best MCC.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def collect(directory, epochs):
    config = json.loads((directory/"experiment_config.json").read_text())
    summary = json.loads((directory/"experiment_summary.json").read_text())
    history = json.loads((directory/"history.json").read_text())
    rows = [r for r in history["eval_history"] if 0 < r["epoch"] <= epochs]
    if not rows or rows[-1]["epoch"] < epochs:
        raise ValueError(f"{directory} has not reached epoch {epochs}")
    manifold = json.loads((directory/"manifold_history.json").read_text())
    schedule = config["optimizer_schedule"]
    steps_per_epoch = schedule["total_optimizer_steps"] / config["num_train_epochs"]
    groups = defaultdict(list)
    for row in manifold["records"]:
        if row["probe"] and row["step"] < steps_per_epoch * epochs:
            groups[row["family"]].append(row)
    family_stats = {}
    for family, records in groups.items():
        family_stats[family] = {
            "probes": len(records),
            "switches": sum(a["active_modes"] != b["active_modes"] for a, b in zip(records, records[1:])),
            "coverage": statistics.mean(r["captured_factor_energy_fraction"] for r in records),
            "horizontal_cost": statistics.mean(
                sum(r["horizontal_costs"][i] for i in r["active_modes"]) for r in records),
        }
    scores = [r["eval_matthews"] for r in rows]
    policy = config["manifold_config"]["policy"]
    layout = config["weight_layout"]
    variant = "stable" if policy == "stable" else ("numeric" if layout == "delta" else "baseline")
    return {
        "directory": str(directory), "variant": variant, "seed": config["seed"], "epochs": epochs,
        "best_mcc": max(scores), "last_mcc": scores[-1],
        "best_epoch": rows[scores.index(max(scores))]["epoch"],
        "zero_mcc_epochs": sum(abs(s) < 1e-12 for s in scores),
        "switches": sum(r["switches"] for r in family_stats.values()),
        "mean_family_coverage": statistics.mean(r["coverage"] for r in family_stats.values()),
        "total_horizontal_cost": sum(r["horizontal_cost"] for r in family_stats.values()),
        # Memory is measured over the complete run, not necessarily this prefix.
        "peak_allocated_gib": summary["training_resources"]["cuda_peak_allocated_bytes"] / 2**30
                              if summary.get("training_resources", {}).get("cuda_peak_allocated_bytes") else None,
        "families": family_stats, "curve": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [collect(path, args.epochs) for path in args.directories]
    groups = defaultdict(list)
    for row in rows:
        groups[row["variant"]].append(row["best_mcc"])
    aggregate = {
        name: {"n": len(values), "mean_best_mcc": statistics.mean(values),
               "sample_sd_best_mcc": statistics.stdev(values) if len(values) > 1 else None}
        for name, values in groups.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"comparison.json").write_text(json.dumps({"runs": rows, "aggregate": aggregate}, indent=2))
    columns = [k for k in rows[0] if k not in ("families", "curve")]
    with (args.output/"comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(aggregate, indent=2))
    print(f"{'Variant':<10} {'Seed':>4} {'Best MCC':>10} {'Last MCC':>10} {'Zero':>5} {'Switches':>9}")
    for row in rows:
        print(f"{row['variant']:<10} {row['seed']:>4} {row['best_mcc']:>10.6f} "
              f"{row['last_mcc']:>10.6f} {row['zero_mcc_epochs']:>5} {row['switches']:>9}")


if __name__ == "__main__":
    main()
