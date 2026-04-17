"""Package entry point for ``python -m tenslora_xs``."""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .cli import main


if __name__ == "__main__":
    main()
