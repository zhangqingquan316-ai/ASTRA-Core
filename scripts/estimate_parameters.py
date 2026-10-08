"""Estimate RoBERTa ASTRA trainable parameter counts without loading weights.

Defaults match the current RoBERTa-large q/k/v/o/FFN rank scan. Counts include
all registered manifold factors, even when the update policy leaves them idle.
"""

import argparse
import csv
import json
from math import prod
from pathlib import Path
import sys


DEFAULT_MODEL = {
    "model_type": "roberta", "num_hidden_layers": 24,
    "num_attention_heads": 16, "hidden_size": 1024, "intermediate_size": 4096,
}
MODES = ("manifold", "additive", "multiplicative", "both")
FAMILIES = ("q", "k", "v", "o", "ffn")


def estimate_parameters(attn_ranks, ffn_ranks, target_families=FAMILIES,
                        tuning_mode="manifold", num_labels=2,
                        multiplicative_num_bases=50, model_config=None):
    config = DEFAULT_MODEL if model_config is None else model_config
    if config.get("model_type") != "roberta":
        raise ValueError("This estimator supports the RoBERTa classification head only.")
    layers, heads, hidden, intermediate = [config[key] for key in
                                          ("num_hidden_layers", "num_attention_heads",
                                           "hidden_size", "intermediate_size")]
    if any(type(value) is not int or value <= 0
           for value in (layers, heads, hidden, intermediate, num_labels)):
        raise ValueError("Model dimensions and num_labels must be positive integers.")
    if hidden % heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads.")
    if tuning_mode not in MODES:
        raise ValueError(f"Unsupported tuning mode: {tuning_mode}")
    if not target_families or len(set(target_families)) != len(target_families):
        raise ValueError("Choose distinct target families.")
    if tuning_mode in ("multiplicative", "both") and (
        type(multiplicative_num_bases) is not int or multiplicative_num_bases <= 0
    ):
        raise ValueError("Multiplicative tuning requires a positive basis count.")
    attention_shape = (layers, heads, hidden // heads, hidden)
    ffn_shape = (2, layers, intermediate, hidden)
    families = {}
    for family in target_families:
        if family not in FAMILIES:
            raise ValueError(f"Unsupported family: {family}")
        dimensions, ranks = ((ffn_shape, ffn_ranks) if family == "ffn"
                             else (attention_shape, attn_ranks))
        if len(ranks) != 4 or any(type(rank) is not int or rank <= 0 or rank > dimension
                                  for rank, dimension in zip(ranks, dimensions)):
            raise ValueError(f"Invalid ranks {ranks} for {family} shape {dimensions}.")
        core = prod(ranks) if tuning_mode in ("manifold", "additive", "both") else 0
        factors = sum(dimension * rank for dimension, rank in zip(dimensions, ranks)) \
            if tuning_mode == "manifold" else 0
        coefficients = 4 * multiplicative_num_bases \
            if tuning_mode in ("multiplicative", "both") else 0
        families[family] = {"core": core, "factors": factors, "coefficients": coefficients,
                            "total": core + factors + coefficients}
    adapter = sum(family["total"] for family in families.values())
    classifier = hidden * hidden + hidden + hidden * num_labels + num_labels
    return {
        "attention_tensor_shape": list(attention_shape),
        "ffn_tensor_shape": list(ffn_shape),
        "families": families,
        "adapter_params": adapter,
        "classifier_trainable_params": classifier,
        "total_trainable_params": adapter + classifier,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attn-ranks", type=int, nargs=4, default=[4, 4, 16, 32])
    parser.add_argument("--ffn-ranks", type=int, nargs=4, default=[2, 4, 64, 64])
    parser.add_argument("--target-families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--tuning-mode", choices=MODES, default="manifold")
    parser.add_argument("--multiplicative-num-bases", type=int, default=50)
    parser.add_argument("--num-labels", type=int, default=2)
    parser.add_argument("--model-config", help="Optional local RoBERTa config.json; defaults to RoBERTa-large.")
    parser.add_argument("--sweep-config", help="Estimate every entry in a sweep JSON; output CSV.")
    parser.add_argument("--output-csv", help="Save sweep estimates as a UTF-8 CSV file.")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text/CSV.")
    args = parser.parse_args()
    if args.output_csv and (not args.sweep_config or args.json):
        parser.error("--output-csv requires --sweep-config and cannot be combined with --json.")
    config = json.loads(Path(args.model_config).read_text(encoding="utf-8")) if args.model_config else None
    specs = json.loads(Path(args.sweep_config).read_text(encoding="utf-8")) if args.sweep_config else [{}]
    if not isinstance(specs, list) or not specs or any(not isinstance(spec, dict) for spec in specs):
        raise ValueError("Sweep config must be a nonempty list of objects.")
    estimates = []
    for index, spec in enumerate(specs, 1):
        spec = {key.replace("-", "_"): value for key, value in spec.items()}
        fields = ("attn_ranks", "ffn_ranks", "target_families", "tuning_mode",
                  "num_labels", "multiplicative_num_bases")
        settings = {key: spec.get(key, getattr(args, key)) for key in fields}
        if spec.get("glue_task") == "stsb" and "num_labels" not in spec:
            settings["num_labels"] = 1
        result = estimate_parameters(**settings, model_config=config)
        result.update(name=spec.get("name", f"experiment_{index}"),
                      attn_ranks=settings["attn_ranks"], ffn_ranks=settings["ffn_ranks"],
                      tuning_mode=settings["tuning_mode"])
        estimates.append(result)
    if args.json:
        print(json.dumps(estimates if args.sweep_config else estimates[0], indent=2))
    elif args.sweep_config:
        fields = ("name", "attn_ranks", "ffn_ranks", "tuning_mode", "adapter_params",
                  "classifier_trainable_params", "total_trainable_params")
        def write_csv(handle):
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(estimates)
        if args.output_csv:
            with Path(args.output_csv).open("w", encoding="utf-8", newline="") as handle:
                write_csv(handle)
            print(f"Saved {len(estimates)} estimates to {args.output_csv}")
        else:
            write_csv(sys.stdout)
    else:
        result = estimates[0]
        print(f"Attention shape: {result['attention_tensor_shape']}; FFN shape: {result['ffn_tensor_shape']}")
        for family, counts in result["families"].items():
            print(f"{family}: core={counts['core']:,}, factors={counts['factors']:,}, "
                  f"coefficients={counts['coefficients']:,}, total={counts['total']:,}")
        for key in ("adapter_params", "classifier_trainable_params", "total_trainable_params"):
            print(f"{key}: {result[key]:,}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(str(error)) from error
