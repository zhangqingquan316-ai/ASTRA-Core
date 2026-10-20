"""Counterfactual selector replay on fixed scores, NOT a training comparison."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astra_core.manifold import ManifoldConfig
from astra_core.stability import StableModeSelector


def summarize(records):
    result = {}
    for family, rows in records.items():
        result[family] = {
            "probes": len(rows),
            "switches": sum(a["modes"] != b["modes"] for a, b in zip(rows, rows[1:])),
            "mean_captured_fraction": sum(r["coverage"] for r in rows)/len(rows),
            "mean_horizontal_cost": sum(r["cost"] for r in rows)/len(rows),
            "mean_active_modes": sum(len(r["modes"]) for r in rows)/len(rows),
        }
    return result


def replay(path):
    source = json.loads(path.read_text(encoding="utf-8"))
    config = ManifoldConfig(**{**source["config"], "policy": "stable", "numerical_mode": "stable"})
    selectors, before, after = {}, defaultdict(list), defaultdict(list)
    for row in source["records"]:
        if not row["probe"]:
            continue
        family = row["family"]
        selector = selectors.setdefault(family, StableModeSelector(config))
        energies, costs, core = row["mode_energies"], row["horizontal_costs"], row["core_energy"]
        selected, _, _ = selector.select(energies, costs, core)
        total = sum(e for e, c in zip(energies, costs) if c > 0)
        for target, modes in ((before, row["active_modes"]), (after, selected)):
            target[family].append({
                "modes": sorted(modes), "cost": sum(costs[i] for i in modes),
                "coverage": sum(energies[i] for i in modes)/total if total else 0.,
            })
    return {"source": str(path), "kind": "fixed-score counterfactual; not retrained",
            "legacy_recorded": summarize(before), "stable_replay": summarize(after)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("histories", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([replay(path) for path in args.histories], indent=2))
