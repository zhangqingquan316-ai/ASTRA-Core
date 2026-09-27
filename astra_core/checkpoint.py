"""Restore an adapter without the original decomposition-cache directory."""

from pathlib import Path

import torch

from .modeling import patch_model_with_tucker, unfreeze_classifier_head
from .training_state import get_trainable_state, load_trainable_state


def load_adapter_checkpoint(model, checkpoint_dir):
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
    basis = torch.load(checkpoint_dir / "adapter_basis.pt", map_location="cpu", weights_only=True)
    state = torch.load(checkpoint_dir / "trainable_state.pt", map_location="cpu", weights_only=True)
    for p in model.parameters():
        p.requires_grad_(False)
    for family in basis:
        patch_model_with_tucker(model, family)
    unfreeze_classifier_head(model)
    expected = get_trainable_state(model)
    if set(state) != set(expected):
        raise ValueError("Adapter parameter names do not match the reconstructed model.")
    for name in expected:
        if expected[name].shape != state[name].shape:
            raise ValueError(f"Checkpoint shape mismatch for {name}.")
    load_trainable_state(model, state)
    return model
