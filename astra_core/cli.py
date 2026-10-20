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
        choices=["additive", "multiplicative", "both", "manifold"],
        default="manifold",
        help="ASTRA-Tangent (manifold, default), or legacy additive/multiplicative/both baselines.",
    )
    parser.add_argument("--manifold-policy", choices=["adaptive", "all", "core", "random"], default="adaptive")
    parser.add_argument("--manifold-probe-interval", type=int, default=10)
    parser.add_argument("--manifold-warmup-steps", type=int, default=20)
    parser.add_argument("--manifold-max-active-modes", type=int, default=2)
    parser.add_argument("--manifold-escape-threshold", type=float, default=0.15)
    parser.add_argument("--manifold-capture-fraction", type=float, default=0.8)
    parser.add_argument("--manifold-cost-power", type=float, default=1.0)
    parser.add_argument("--manifold-damping", type=float, default=1e-3)
    parser.add_argument("--manifold-factor-lr-scale", type=float, default=1.0)
    parser.add_argument("--manifold-max-factor-step", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--multiplicative-num-bases",
        "--factor-tuning-params",
        dest="multiplicative_num_bases",
        type=int,
        default=50,
        help="Number of fixed basis matrices for each Tucker-core mode transform used in multiplicative tuning.",
    )
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max length.")
    parser.add_argument("--learning-rate", "--encoder-learning-rate", type=float, default=1e-3,
                        help="Encoder adapter learning rate; also the head rate unless overridden.")
    parser.add_argument("--classifier-learning-rate", "--head-learning-rate", type=float, default=None,
                        help="Separate classifier rate in manifold mode; defaults to the encoder rate.")
    parser.add_argument("--warmup-ratio", type=float, default=0.0,
                        help="Fraction of optimizer steps used for learning-rate warmup.")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Explicit learning-rate warmup steps; use with warmup-ratio=0.")
    parser.add_argument("--lr-scheduler-type", choices=["linear", "cosine", "constant_with_warmup"],
                        default="linear", help="Learning-rate schedule after warmup.")
    parser.add_argument("--gpu-memory-fraction", type=float, default=1.0,
                        help="Per-process CUDA allocator limit as a fraction of GPU memory.")
    parser.add_argument(
        "--weight-layout",
        choices=["residual", "two_matrix"],
        default="residual",
        help="residual stores W0 - alpha*X0 once. two_matrix stores W0 and X0 separately.",
    )
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
