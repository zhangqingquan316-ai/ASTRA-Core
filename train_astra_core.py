"""Convenience wrapper for running ASTRA-Core from the repository root."""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from astra_core.cli import main


if __name__ == "__main__":
    main()
