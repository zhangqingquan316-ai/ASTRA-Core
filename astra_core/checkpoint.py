"""Restore an adapter without the original decomposition-cache directory."""

from pathlib import Path
import json
import warnings

import torch

from .modeling import patch_model_with_tucker, unfreeze_classifier_head
from .training_state import get_trainable_state, load_trainable_state


def load_adapter_checkpoint(model, checkpoint_dir, weight_layout=None):
    """Patch a fresh CPU base model, then restore all adapter/head parameters.

    The caller must supply the same pretrained weights and classifier geometry.
    Move to the desired device *after* loading. This restores inference state;
    optimizer/scheduler/RNG state is not contained in final adapter artifacts.
    """
    if any(name.startswith("astra_core_shared_state_") for name, _ in model.named_modules()):
        raise ValueError("Expected an unpatched base model.")
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("Load the adapter into a CPU model before moving it to the accelerator.")
    checkpoint_dir = Path(checkpoint_dir)
    if weight_layout is None:
        for metadata in (checkpoint_dir / "training_config.json",
                         checkpoint_dir.parent / "experiment_config.json"):
            if metadata.exists():
                weight_layout = json.loads(metadata.read_text(encoding="utf-8")).get("weight_layout")
                if weight_layout:
                    break
    if weight_layout is None:
        warnings.warn("Checkpoint has no weight_layout metadata; assuming legacy two_matrix. "
                      "Pass weight_layout explicitly for metadata-free residual artifacts.", UserWarning)
        weight_layout = "two_matrix"
    basis = torch.load(checkpoint_dir / "adapter_basis.pt", map_location="cpu", weights_only=True)
    state = torch.load(checkpoint_dir / "trainable_state.pt", map_location="cpu", weights_only=True)
    for p in model.parameters():
        p.requires_grad_(False)
    for family in basis:
        patch_model_with_tucker(model, family, weight_layout=weight_layout)
    unfreeze_classifier_head(model)
    expected = get_trainable_state(model)
    if set(state) != set(expected):
        raise ValueError("Adapter parameter names do not match the reconstructed model.")
    for name in expected:
        if expected[name].shape != state[name].shape:
            raise ValueError(f"Checkpoint shape mismatch for {name}.")
    load_trainable_state(model, state)
    return model
