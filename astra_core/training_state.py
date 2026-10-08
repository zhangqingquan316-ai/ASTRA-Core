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


class OptimizerScheduleCallback(TrainerCallback):
    """Record the Trainer's actual optimizer-step budget and LR warmup."""

    def __init__(self, experiment_args):
        self.experiment_args = experiment_args

    def on_train_begin(self, args, state, control, **kwargs):
        schedule = {
            "lr_scheduler_type": getattr(args.lr_scheduler_type, "value", args.lr_scheduler_type),
            "warmup_ratio": args.warmup_ratio,
            "configured_warmup_steps": args.warmup_steps,
            "actual_warmup_steps": args.get_warmup_steps(state.max_steps),
            "total_optimizer_steps": state.max_steps,
            "peak_learning_rate": args.learning_rate,
            "encoder_peak_learning_rate": self.experiment_args.learning_rate,
            "classifier_peak_learning_rate": (getattr(self.experiment_args, "classifier_learning_rate", None)
                                              or self.experiment_args.learning_rate),
        }
        optimizer = kwargs.get("optimizer")
        if optimizer is not None:
            schedule["parameter_groups"] = [
                {"name": group.get("name", str(index)),
                 "peak_learning_rate": group.get("initial_lr", group["lr"]),
                 "parameters": sum(p.numel() for p in group["params"]),
                 "weight_decay": group["weight_decay"]}
                for index, group in enumerate(optimizer.param_groups)
            ]
        self.experiment_args._optimizer_schedule = schedule
        print(f"Optimizer schedule: {schedule}", flush=True)
        return control


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

