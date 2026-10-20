"""Check whole-model zero-anchor equivalence on local RoBERTa/CoLA, under AMP."""
import argparse
import gc
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from astra_core.modeling import (get_or_create_decomposition_cache_entry,
                                 build_runtime_family_from_decomposition_cache,
                                 patch_model_with_tucker, unfreeze_classifier_head)
from astra_core.stability import incremental_polar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(project/"local_models/roberta-large", local_files_only=True)
    rows = load_from_disk(str(project/"local_datasets/glue_cola"))["train"][:8]
    inputs = tokenizer(rows["sentence"], padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    inputs["labels"] = torch.tensor(rows["label"], device="cuda")
    reference, head_reference, report = None, None, {}
    for variant in ("pretrained", "delta"):
        set_seed(42)
        model = AutoModelForSequenceClassification.from_pretrained(
            project/"local_models/roberta-large", num_labels=2, local_files_only=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if variant == "delta":
            for family in ("q", "k", "v", "o", "ffn"):
                ranks = (2, 16, 64, 64) if family == "ffn" else (24, 4, 16, 32)
                entry = get_or_create_decomposition_cache_entry(
                    model, str(project/"local_models/roberta-large"),
                    project/"tucker_hooi_cache/roberta_large", family, ranks, 16, {})
                runtime = build_runtime_family_from_decomposition_cache(entry, 1., "manifold", 50, 42)
                patch_model_with_tucker(model, runtime, weight_layout="delta")
        unfreeze_classifier_head(model)
        model.cuda().eval()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(**inputs)
        output.loss.backward()
        logits = output.logits.detach().float().cpu()
        head = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
                if n.startswith("classifier.") and p.grad is not None}
        if reference is None:
            reference, head_reference = logits, head
        row = {
            "loss": float(output.loss), "logits_bitwise_equal": torch.equal(logits, reference),
            "max_logit_error": float((logits-reference).abs().max()),
            "head_gradients_bitwise_equal": all(torch.equal(v, head_reference[n]) for n, v in head.items()),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        if variant == "delta":
            family = model.astra_core_shared_state_k
            u = family.manifold_factors()[1].detach()
            direction = torch.randn_like(u).double()
            direction -= u.double() @ (u.double().T @ direction)
            direction /= direction.norm()
            row["retraction"] = [
                {"requested": size, "actual": float((incremental_polar(u, size*direction).double()-u).norm())}
                for size in (0., 1e-12, 1e-9, 1e-7, 1e-5, 1e-3)]
            row["all_adapter_gradients_finite"] = all(
                torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
            assert row["logits_bitwise_equal"] and row["head_gradients_bitwise_equal"]
            assert row["all_adapter_gradients_finite"] and row["retraction"][0]["actual"] == 0
        report[variant] = row
        del model, output, head
        gc.collect()
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
