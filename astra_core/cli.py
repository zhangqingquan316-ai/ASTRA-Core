"""Thin CLI wrapper that keeps argument parsing separate from experiment logic."""

import argparse
import os

from .constants import TASK_CONFIGS
from .experiment import run_from_cli


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ASTRA-Core family-wise Tucker adapters with Tucker/HOOI decomposition caches on local GLUE tasks."
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("TUCKER_MODEL_PATH", "./local_models/roberta-large"),
        help="Local Hugging Face model directory.",
    )
    parser.add_argument(
        "--dataset-path",
        default=os.getenv("TUCKER_DATASET_PATH"),
        help="Local datasets.save_to_disk() directory. Defaults to ./local_datasets/glue_<task>.",
    )
    parser.add_argument(
        "--glue-task",
        choices=sorted(TASK_CONFIGS),
        default="sst2",
        help="Which GLUE task to run.",
    )
    parser.add_argument(
        "--target-families",
        nargs="+",
        choices=["q", "k", "v", "o", "ffn"],
        default=["q", "k", "v", "o"],
        help="Which families to adapt. Default: q k v o",
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help="Backward-compatible aliases such as query key value attention.output.dense ffn.",
    )
    parser.add_argument(
        "--attn-ranks",
        "--ranks",
        dest="attn_ranks",
        type=int,
        nargs=4,
        metavar=("R_LAYER", "R_HEAD", "R_HEAD_DIM", "R_HIDDEN"),
        default=(4, 4, 16, 32),
        help="Attention Tucker ranks for [layer, head, head_dim, hidden].",
    )
    parser.add_argument(
        "--ffn-ranks",
        type=int,
        nargs=4,
        metavar=("R_FFN_STAGE", "R_LAYER", "R_M", "R_N"),
        default=(2, 4, 64, 64),
        help="FFN Tucker ranks for [ffn_stage, layer, m, n].",
    )
    parser.add_argument(
        "--attn-alpha",
        type=float,
        default=1.0,
        help="Attention adapter scaling coefficient.",
    )
    parser.add_argument(
        "--ffn-alpha",
        type=float,
        default=1.0,
        help="FFN adapter scaling coefficient.",
    )
    parser.add_argument(
        "--tuning-mode",
        choices=["additive", "multiplicative", "both"],
        default="additive",
        help="Which ASTRA SFT variant to train: additive core update (ASTRA-Core), multiplicative core-mode transform (ASTRA-Mode), or both (ASTRA-Hybrid).",
    )
    parser.add_argument(
        "--multiplicative-num-bases",
        "--factor-tuning-params",
        dest="multiplicative_num_bases",
        type=int,
        default=50,
        help="Number of fixed basis matrices for each Tucker-core mode transform used in multiplicative tuning.",
    )
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max length.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=32,
        help="Train batch size per device.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=32,
        help="Eval batch size per device.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=3.0,
        help="Number of training epochs.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional name for a single experiment run.",
    )
    parser.add_argument(
        "--sweep-config",
        default=None,
        help="Path to a JSON file that lists multiple experiment settings to run sequentially.",
    )
    parser.add_argument(
        "--sweep-output-dir",
        default=None,
        help="Optional directory for storing sweep experiment folders and the final comparison plot.",
    )
    parser.add_argument(
        "--decomposition-cache-dir",
        "--hosvd-cache-dir",
        dest="decomposition_cache_dir",
        default=None,
        help="Directory for storing Tucker/HOOI decomposition caches. '--hosvd-cache-dir' is kept as a backward-compatible alias.",
    )
    return parser.parse_args()


def main():
    """Parse CLI arguments and hand off to the experiment runner."""
    run_from_cli(parse_args())
