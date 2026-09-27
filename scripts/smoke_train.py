"""Offline end-to-end CLI smoke run; creates a tiny random BERT and toy SST-2.

Run after `pip install -e .`: python scripts/smoke_train.py
These generated labels/weights are a software check, not a GLUE benchmark.
"""

import json
from pathlib import Path
import subprocess
import sys

from datasets import Dataset, DatasetDict
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from astra_core.checkpoint import load_adapter_checkpoint


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    root = Path("runs/offline_smoke").resolve()
    model_path, dataset_path = root / "model", root / "dataset"
    model_path.mkdir(parents=True, exist_ok=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "good", "bad", "movie", "very"]
    vocab_path = model_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path))
    tokenizer.save_pretrained(model_path)
    config = BertConfig(vocab_size=len(vocab), hidden_size=8, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=12, num_labels=2)
    model = BertForSequenceClassification(config)
    model.save_pretrained(model_path)
    data = Dataset.from_dict({"sentence": ["good movie", "bad movie"] * 4, "label": [1, 0] * 4})
    DatasetDict(train=data, validation=data.select(range(4))).save_to_disk(dataset_path)
    sweep = [{"name": p, "tuning_mode": "manifold", "manifold_policy": p}
             for p in ("adaptive", "core", "all", "random")]
    sweep_path = root / "smoke_sweep.json"
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")
    subprocess.run([
        sys.executable, "-m", "astra_core", "--model-path", str(model_path),
        "--dataset-path", str(dataset_path), "--target-families", "q", "o", "ffn",
        "--attn-ranks", "1", "1", "2", "2", "--ffn-ranks", "1", "1", "2", "2",
        "--num-train-epochs", "1", "--per-device-train-batch-size", "2",
        "--gradient-accumulation-steps", "2", "--manifold-warmup-steps", "0",
        "--manifold-probe-interval", "1", "--manifold-escape-threshold", "0",
        "--sweep-config", str(sweep_path), "--sweep-output-dir", str(root / "sweep"),
        "--decomposition-cache-dir", str(root / "cache"),
    ], check=True)
    for policy in ("adaptive", "core", "all", "random"):
        checkpoint = root / "sweep" / policy / "final"
        restored = load_adapter_checkpoint(BertForSequenceClassification.from_pretrained(model_path), checkpoint)
        restored.eval()
        with torch.no_grad():
            logits = restored(**tokenizer(["good movie", "bad movie"], return_tensors="pt")).logits
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"Nonfinite restored logits: {policy}")
    print(f"Offline CLI sweep and checkpoint reload passed: {root}")


if __name__ == "__main__":
    main()
