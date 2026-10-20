"""Budgeted horizontal Tucker updates (ASTRA-Tangent).

Only the selector is proposed as a research mechanism. Horizontal projection,
core-Gram preconditioning and polar retraction are standard geometric tools.
"""

from dataclasses import asdict, dataclass
import math

import torch
from .stability import (StableModeSelector, capped_factor_step, incremental_polar,
                        normalized_tangent_statistics)


@dataclass
class ManifoldConfig:
    policy: str = "adaptive"
    probe_interval: int = 10
    warmup_steps: int = 20
    max_active_modes: int = 2
    escape_threshold: float = 0.15
    capture_fraction: float = 0.8
    cost_power: float = 1.0
    damping: float = 1e-3
    factor_lr_scale: float = 1.0
    max_factor_step: float = 0.05
    seed: int = 42
    numerical_mode: str = "legacy"
    score_ema: float = 0.9
    escape_hysteresis: float = 0.03
    coverage_hysteresis: float = 0.03
    confirm_probes: int = 3
    decision_tolerance: float = 1e-3
    orthogonality_tolerance: float = 1e-5

    def __post_init__(self):
        if self.policy not in {"stable", "adaptive", "all", "core", "random"}:
            raise ValueError("Unknown manifold policy.")
        if self.numerical_mode not in {"legacy", "stable"}:
            raise ValueError("Unknown numerical mode.")
        for name in ("probe_interval", "warmup_steps", "max_active_modes", "seed", "confirm_probes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if self.probe_interval < 1:
            raise ValueError("probe_interval must be positive.")
        if self.confirm_probes < 1:
            raise ValueError("confirm_probes must be positive.")
        for name in ("escape_threshold", "capture_fraction", "cost_power", "damping",
                     "factor_lr_scale", "max_factor_step", "score_ema", "escape_hysteresis",
                     "coverage_hysteresis", "decision_tolerance", "orthogonality_tolerance"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")
        if not 0 <= self.escape_threshold <= 1 or not 0 < self.capture_fraction <= 1:
            raise ValueError("Invalid escape threshold or capture fraction.")
        if self.cost_power < 0 or min(self.damping, self.factor_lr_scale, self.max_factor_step) <= 0:
            raise ValueError("Damping, factor LR scale and step cap must be positive; cost_power >= 0.")
        if not 0 <= self.score_ema < 1:
            raise ValueError("score_ema must be in [0, 1).")
        if not all(0 <= v < 1 for v in (self.escape_hysteresis, self.coverage_hysteresis,
                                       self.decision_tolerance)):
            raise ValueError("Hysteresis and decision tolerance must be in [0, 1).")
        if self.orthogonality_tolerance <= 0:
            raise ValueError("orthogonality_tolerance must be positive.")

    @classmethod
    def from_args(cls, args):
        values = {name: getattr(args, f"manifold_{name}", field.default)
                  for name, field in cls.__dataclass_fields__.items() if name != "seed"}
        return cls(**values, seed=getattr(args, "seed", 42))


def horizontal_direction(factor, gradient, core, mode, damping):
    """Return D = (I-UU^T) grad_U (C_(k) C_(k)^T + lambda I)^-1.

    No ambient-sized gradient or n-by-n projector is materialized. The
    returned nonnegative pairing <grad_U,D> estimates first-order decrease.
    FP32 linalg under AMP, FP64 preserved for numerical verification.
    """
    dtype = torch.float64 if factor.dtype == torch.float64 else torch.float32
    u, g, c = factor.to(dtype), gradient.to(dtype), core.to(dtype)
    if u.shape[0] == u.shape[1]:
        return torch.zeros_like(u), 0.0
    h = g - u @ (u.T @ g)
    unfolding = c.movedim(mode, 0).reshape(c.shape[mode], -1)
    gram = unfolding @ unfolding.T
    ridge = damping * gram.diagonal().mean().clamp_min(torch.finfo(dtype).eps)
    regularized = gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=dtype)
    direction = torch.linalg.solve(regularized, h.T).T
    return direction, max(0.0, float((h * direction).sum()))


def polar_retraction(factor, step):
    """Polar(U + step), an orthogonally equivariant Stiefel retraction."""
    left, _, right = torch.linalg.svd(factor + step, full_matrices=False)
    return left @ right


def select_modes(energies, costs, core_energy, config):
    """Budgeted greedy selection after a family-level escape-energy test.

    Costs are horizontal degrees of freedom r(n-r), a proxy, not measured FLOPs.
    A full-rank square factor has no horizontal directions and is excluded.
    """
    candidates = [i for i, (e, c) in enumerate(zip(energies, costs)) if c > 0 and e > 0]
    total = sum(energies[i] for i in candidates)
    escape = total / (core_energy + total) if core_energy + total > 0 else 0.0
    if escape < config.escape_threshold or not total or not config.max_active_modes:
        return [], escape
    candidates.sort(key=lambda i: (-energies[i] / costs[i] ** config.cost_power, i))
    selected, captured = [], 0.0
    for mode in candidates[:config.max_active_modes]:
        selected.append(mode)
        captured += energies[mode]
        if captured >= config.capture_fraction * total:
            break
    return selected, escape


def mode_scores(energies, costs, cost_power):
    """s_k = e_k / [r_k (n_k - r_k)]^p. Square factors and zero energies score 0."""
    scores = []
    for energy, cost in zip(energies, costs):
        if cost > 0 and energy > 0:
            scores.append(energy / cost ** cost_power)
        else:
            scores.append(0.0)
    return scores


class TangentAdamW(torch.optim.AdamW):
    """AdamW for cores/heads; stateless preconditioned horizontal factor steps.

    Directions are computed from the same pre-update state as the core gradient.
    Factors never receive Adam momentum or weight decay. This is a hybrid
    optimizer, not a claim of fully gauge-invariant Riemannian Adam.
    """

    def __init__(self, model, lr=1e-3, weight_decay=0.01, config=None, classifier_lr=None):
        if classifier_lr is not None and (not math.isfinite(classifier_lr) or classifier_lr <= 0):
            raise ValueError("classifier_lr must be finite and positive.")
        self.config = config or ManifoldConfig()
        self.families = [m for m in model.modules() if getattr(m, "is_manifold", False)]
        if not self.families:
            raise ValueError("TangentAdamW requires at least one manifold Tucker family.")
        factors = [p for family in self.families for p in family.manifold_factors()]
        if any(p.dtype not in (torch.float32, torch.float64) for p in factors):
            raise ValueError("Keep manifold parameters in FP32/FP64; use autocast for mixed precision.")
        factor_ids = {id(p) for p in factors}
        ordinary = [(name, p) for name, p in model.named_parameters()
                    if p.requires_grad and id(p) not in factor_ids]
        heads = [(name, p) for name, p in ordinary
                 if name.startswith(("classifier.", "score."))] if classifier_lr is not None else []
        head_ids = {id(p) for _, p in heads}
        if classifier_lr is not None and not heads:
            raise ValueError("A separate classifier rate requires trainable classifier.* or score.* parameters.")
        # Keep factors at index 1: geometric steps below use this group's scheduled LR.
        groups = [
            {"params": [p for _, p in ordinary if id(p) not in head_ids],
             "weight_decay": weight_decay, "name": "encoder_core"},
            {"params": factors, "weight_decay": 0.0,
             "lr": lr * self.config.factor_lr_scale, "name": "encoder_factors"},
        ]
        if heads:
            groups.append({"params": [p for _, p in heads], "weight_decay": weight_decay,
                           "lr": classifier_lr, "name": "classifier"})
        super().__init__(groups, lr=lr)
        self.completed_steps = 0
        self.skipped_steps = 0
        self.history = []
        self.selectors = [StableModeSelector(self.config) for _ in self.families]
        self._prepare_forward()

    def _is_probe(self):
        c = self.config
        return self.completed_steps >= c.warmup_steps and (
            (self.completed_steps - c.warmup_steps) % c.probe_interval == 0)

    def _prepare_forward(self):
        c = self.config
        for family in self.families:
            if self.completed_steps < c.warmup_steps or c.policy == "core":
                family.active_modes = set()
                family.probe_factors = False
            elif c.policy == "all":
                family.active_modes = {i for i, p in enumerate(family.manifold_factors())
                                       if p.shape[0] > p.shape[1]}
                family.probe_factors = False
            else:
                family.probe_factors = self._is_probe()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        params = [p for group in self.param_groups for p in group["params"]]
        if any(not torch.isfinite(p.grad).all() for p in params if p.grad is not None):
            self.skipped_steps += 1
            return loss

        c = self.config
        proposals, records = [], []
        for family_index, family in enumerate(self.families):
            factors = family.manifold_factors()
            core = family.get_updated_core(family.delta_core.dtype).detach()
            core_energy = 0.0
            directions, energies = {}, [0.0] * len(factors)
            costs = [p.shape[1] * (p.shape[0] - p.shape[1]) for p in factors]
            gradient_scale, diagnostics = None, {}
            if c.numerical_mode == "stable":
                directions, energies, core_energy, gradient_scale = normalized_tangent_statistics(
                    factors, core, family.delta_core.grad, c.damping)
            else:
                if family.delta_core.grad is not None:
                    core_energy = float(family.delta_core.grad.double().square().sum())
                for i, p in enumerate(factors):
                    if p.grad is not None and costs[i] > 0:
                        directions[i], energies[i] = horizontal_direction(p, p.grad, core, i, c.damping)
            total = sum(energies)
            escape = total / (core_energy + total) if core_energy + total else 0.0
            if family.probe_factors:
                if c.policy == "stable":
                    selected, escape, diagnostics = self.selectors[family_index].select(energies, costs, core_energy)
                else:
                    selected, escape = select_modes(energies, costs, core_energy, c)
                if c.policy == "random":
                    # Same probe-dependent cardinality as adaptive: isolates routing quality.
                    eligible = [i for i, cost in enumerate(costs) if cost > 0]
                    generator = torch.Generator().manual_seed(c.seed + self.completed_steps * 1009 + family_index)
                    order = torch.randperm(len(eligible), generator=generator).tolist()
                    selected = [eligible[i] for i in order[:len(selected)]]
                family.active_modes = set(selected)

            step_norms, actual_step_norms = {}, {}
            for i in sorted(family.active_modes):
                if i not in directions:
                    continue
                step = (capped_factor_step(directions[i], self.param_groups[1]["lr"], c.max_factor_step)
                        if c.numerical_mode == "stable" else
                        -self.param_groups[1]["lr"] * directions[i])
                norm = float(torch.linalg.vector_norm(step))
                if norm > c.max_factor_step:
                    step.mul_(c.max_factor_step / norm)
                candidate = (incremental_polar(factors[i], step, c.orthogonality_tolerance)
                             if c.numerical_mode == "stable" else
                             polar_retraction(factors[i].to(step.dtype), step))
                if not torch.isfinite(candidate).all():
                    raise FloatingPointError("Nonfinite factor proposal; no optimizer update applied.")
                proposals.append((factors[i], candidate))
                step_norms[str(i)] = min(norm, c.max_factor_step)
                actual_step_norms[str(i)] = float(torch.linalg.vector_norm(
                    candidate.double() - factors[i].double()))
            if self._is_probe() or self.completed_steps == 0:
                scores = mode_scores(energies, costs, c.cost_power)
                decision_applied = bool(family.probe_factors)
                core_only = decision_applied and not family.active_modes
                records.append({
                    "step": self.completed_steps, "family": family.family_name,
                    "policy": c.policy, "probe": family.probe_factors,
                    "numerical_mode": c.numerical_mode,
                    "gradient_scale": gradient_scale,
                    "energy_units": "gradient_scale_squared" if gradient_scale is not None else "raw",
                    "core_energy": core_energy, "mode_energies": energies,
                    "e0": core_energy, "e_k": list(energies),
                    "escape_ratio": escape, "rho": escape,
                    "tau": c.escape_threshold,
                    "s_k": scores,
                    "score_order": sorted(
                        (i for i, score in enumerate(scores) if score > 0),
                        key=lambda i: (-scores[i], i)),
                    "gamma": c.capture_fraction,
                    "max_active_modes": c.max_active_modes,
                    "cost_power": c.cost_power,
                    "decision_applied": decision_applied,
                    "core_only": core_only,
                    "active_modes": sorted(family.active_modes),
                    "captured_factor_energy_fraction": (
                        sum(energies[i] for i in family.active_modes) / total if total else 0.0),
                    "estimated_missed_tangent_fraction": (
                        sum(e for i, e in enumerate(energies) if i not in family.active_modes)
                        / (core_energy + total) if core_energy + total else 0.0),
                    "horizontal_costs": costs, "factor_step_norms": step_norms,
                    "actual_factor_step_norms": actual_step_norms,
                    "active_factor_params": sum(factors[i].numel() for i in family.active_modes),
                    "orthogonality_error": max(float(torch.linalg.matrix_norm(
                        p.T @ p - torch.eye(p.shape[1], device=p.device, dtype=p.dtype))) for p in factors),
                    **diagnostics,
                })

        # Suppress AdamW for factors (including its momentum and decoupled decay).
        saved_grads = [p.grad for p in self.param_groups[1]["params"]]
        for p in self.param_groups[1]["params"]:
            p.grad = None
        try:
            super().step()
        finally:
            for p, grad in zip(self.param_groups[1]["params"], saved_grads):
                p.grad = grad
        for p, candidate in proposals:
            p.copy_(candidate.to(p.dtype))
        self.history.extend(records)
        self.completed_steps += 1
        self._prepare_forward()
        return loss

    def state_dict(self):
        result = super().state_dict()
        result["tangent_controller"] = {
            "config": asdict(self.config), "completed_steps": self.completed_steps,
            "skipped_steps": self.skipped_steps, "history": self.history,
            "active_modes": [sorted(f.active_modes) for f in self.families],
            "family_names": [f.family_name for f in self.families],
            "selectors": [s.state_dict() for s in self.selectors],
        }
        return result

    def load_state_dict(self, state_dict):
        payload = dict(state_dict)
        controller = payload.pop("tangent_controller")
        if controller["family_names"] != [f.family_name for f in self.families]:
            raise ValueError("Optimizer checkpoint family order does not match the model.")
        super().load_state_dict(payload)
        self.config = ManifoldConfig(**controller["config"])
        self.completed_steps = controller["completed_steps"]
        self.skipped_steps = controller["skipped_steps"]
        self.history = list(controller["history"])
        self.selectors = [StableModeSelector(self.config) for _ in self.families]
        states = controller.get("selectors")
        if states is not None:
            if len(states) != len(self.selectors):
                raise ValueError("Selector checkpoint family count does not match.")
            for selector, state in zip(self.selectors, states):
                selector.load_state_dict(state)
        for family, modes in zip(self.families, controller["active_modes"]):
            family.active_modes = set(modes)
        self._prepare_forward()
