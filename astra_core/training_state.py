"""Helpers for tracking and restoring the trainable part of the model."""

import torch
from transformers import TrainerCallback


def get_trainable_state(model):
    """Snapshot only the parameters that are actually trained in this workflow."""
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_trainable_state(model, state_dict):
    """Restore only the trainable parameters into an already patched model."""
    if state_dict is None:
        return

    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state_dict.items():
            if name in named_params:
                param = named_params[name]
                param.copy_(tensor.to(device=param.device, dtype=param.dtype))


class BestModelByMetricCallback(TrainerCallback):
    """Track the best trainable state according to the main validation metric."""

    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.best_metric = float("-inf")
        self.best_epoch = None
        self.best_state = None

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None or metrics is None or self.metric_name not in metrics:
            return control

        current_metric = float(metrics[self.metric_name])
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_epoch = float(state.epoch) if state.epoch is not None else None
            self.best_state = get_trainable_state(model)
            print(
                f"New best model found: {self.metric_name}={self.best_metric:.6f} "
                f"at epoch={self.best_epoch}"
            )
        return control

