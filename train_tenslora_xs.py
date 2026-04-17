"""Convenience wrapper for running TensLoRA-XS from the repository root."""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from tenslora_xs.cli import main


if __name__ == "__main__":
    main()
