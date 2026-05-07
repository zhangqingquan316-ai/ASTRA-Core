"""Package entry point for ``python -m astra_core``."""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .cli import main


if __name__ == "__main__":
    main()
