"""Download a GLUE task and save it to disk for the local-only training workflow."""

import argparse
from pathlib import Path
import shutil

from datasets import load_dataset


SUPPORTED_TASKS = ["sst2", "mrpc", "cola", "qnli", "rte", "stsb"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a GLUE task and save it with datasets.save_to_disk()."
    )
    parser.add_argument(
        "--task",
        choices=SUPPORTED_TASKS,
        required=True,
        help="Which GLUE task to download.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the dataset will be saved. Defaults to ./local_datasets/glue_<task>.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output directory if it already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir or f"./local_datasets/glue_{args.task}")

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir.resolve()}. "
                "Pass --force if you want to overwrite it."
            )
        shutil.rmtree(output_dir)
    else:
        output_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading GLUE task '{args.task}' from Hugging Face datasets...")
    dataset = load_dataset("glue", args.task, cache_dir=args.cache_dir)

    print(f"Saving dataset to: {output_dir.resolve()}")
    dataset.save_to_disk(str(output_dir))

    split_sizes = {split: dataset[split].num_rows for split in dataset.keys()}
    print(f"Done. Saved splits: {split_sizes}")


if __name__ == "__main__":
    main()
