"""Scale-free probe statistics, stateful coverage selection, and small-step geometry.

The statistics contain no learning rate. Normalize gradients *before* quadratic
forms, and normalize each full probe before temporal averaging. A common loss
scale therefore does not affect decisions (up to finite input precision).
"""
from itertools import combinations
import math

import torch


def capped_factor_step(direction, learning_rate, cap):
    """Apply LR and norm cap without overflowing a squared norm/product."""
    if not torch.isfinite(direction).all():
        raise FloatingPointError("Nonfinite tangent direction.")
    largest = float(direction.abs().max())
    if learning_rate == 0 or largest == 0:
        return torch.zeros_like(direction)
    unit = direction / largest
    unit_norm = float(torch.linalg.vector_norm(unit))
    if largest > cap / (learning_rate * unit_norm):
        return unit * (-cap / unit_norm)
    return direction * (-learning_rate)


def normalized_tangent_statistics(factors, core, core_gradient, damping):
    """Return actual FP64 directions and energies divided by one gradient scale².

    One common scale includes the core and every available factor gradient.
    Only full probes are suitable for comparing these statistics over time.
    FP64 small Gram solves/projections also avoid FP32 squared-gradient underflow.
    """
    gradients = [core_gradient] + [p.grad for p in factors]
    scale = max((float(g.detach().abs().max()) for g in gradients if g is not None), default=0.)
    if not math.isfinite(scale):
        raise ValueError("Nonfinite gradient scale.")
    if scale == 0:
        return {}, [0.] * len(factors), 0., 0.
    core_energy = float((core_gradient.double() / scale).square().sum()) if core_gradient is not None else 0.
    c = core.double()
    directions, energies = {}, [0.] * len(factors)
    for mode, factor in enumerate(factors):
        if factor.grad is None or factor.shape[0] == factor.shape[1]:
            continue
        u, g = factor.double(), factor.grad.double() / scale
        h = g - u @ (u.T @ g)
        unfolding = c.movedim(mode, 0).reshape(c.shape[mode], -1)
        gram = unfolding @ unfolding.T
        # Keep the legacy regularization floor: dtype precision is not a new
        # optimization hyperparameter. Only evaluation precision changes.
        floor = torch.finfo(core.dtype).eps
        ridge = damping * gram.diagonal().mean().clamp_min(floor)
        regularized = gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        direction = torch.linalg.solve(regularized, h.T).T
        energies[mode] = max(0., float((h * direction).sum()))
        directions[mode] = direction * scale
    return directions, energies, core_energy, scale


def incremental_polar(factor, step, orthogonality_tolerance=1e-5):
    """Polar retraction with a cancellation-free increment near the identity.

    For exactly orthonormal U this equals polar(U+step). For stored FP32 U,
    tolerate its existing defect below the explicit tolerance, rather than
    repeatedly projecting U during a sub-ULP step. Zero steps are exact no-ops.
    """
    if not torch.count_nonzero(step):
        return factor.clone()
    u, s = factor.double(), step.double()
    identity = torch.eye(u.shape[1], dtype=u.dtype, device=u.device)
    defect = u.T @ u - identity
    x = u + s
    if torch.linalg.matrix_norm(defect) > orthogonality_tolerance:
        values, vectors = torch.linalg.eigh(x.T @ x)
        if torch.any(values <= 0):
            raise ValueError("Retraction input lost column rank.")
        return (x @ ((vectors * values.rsqrt()) @ vectors.T)).to(factor.dtype)
    change = u.T @ s + s.T @ u + s.T @ s
    values, vectors = torch.linalg.eigh(change)
    if torch.any(values <= -1):
        raise ValueError("Retraction Gram is not positive definite.")
    root = torch.sqrt(1 + values)
    correction = (vectors * (-values / (root * (1 + root)))) @ vectors.T
    return (x + x @ correction).to(factor.dtype)


def energy_shares(energies, core_energy, costs):
    """Stable normalization even when finite input energies are 1e±300."""
    values = [core_energy] + [e if cost > 0 else 0. for e, cost in zip(energies, costs)]
    if len(energies) != len(costs) or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Expected finite nonnegative energies and matching costs.")
    scale = max(values, default=0.)
    if scale == 0:
        return [0.] * len(values)
    scaled = [v / scale for v in values]
    total = math.fsum(scaled)
    return [v / total for v in scaled]


def coverage_subset(shares, costs, max_modes, target, cost_power, tolerance):
    """Minimum proxy cost at target coverage; maximum coverage if infeasible.

    There are only four modes per family. Exact subset enumeration avoids the
    inconsistent density-greedy + cardinality + coverage stopping rule.
    Near ties use a fixed tolerance in *normalized share*, then cost/index order.
    """
    eligible = [i for i, (p, c) in enumerate(zip(shares, costs)) if c > 0 and p > 0]
    if not eligible or max_modes == 0:
        return (), False
    total = math.fsum(shares[i] for i in eligible)
    fractions = [v / total for v in shares]
    choices = [s for n in range(1, min(max_modes, len(eligible)) + 1)
               for s in combinations(eligible, n)]
    largest_cost = max(costs[i] for i in eligible)
    cost = lambda s: math.fsum((costs[i] / largest_cost) ** cost_power for i in s)
    gain = lambda s: math.fsum(fractions[i] for i in s)
    feasible = [s for s in choices if gain(s) >= target]
    if feasible:
        # Favor cost first; stable index ordering resolves exact/near cost ties.
        minimum = min(cost(s) for s in feasible)
        tied = [s for s in feasible if cost(s) <= minimum + tolerance]
        return min(tied, key=lambda s: (len(s), s)), True
    best = max(gain(s) for s in choices)
    tied = [s for s in choices if gain(s) >= best - tolerance]
    return min(tied, key=lambda s: (cost(s), len(s), s)), False


class StableModeSelector:
    """EMA shares + escape hysteresis + coverage retention + switch confirmation."""
    def __init__(self, config):
        self.config = config
        self.ema = None
        self.active = ()
        self.pending = None
        self.pending_count = 0
        self.gate_open = False
        self.probes = 0

    def state_dict(self):
        return {"ema": self.ema, "active": list(self.active),
                "pending": None if self.pending is None else list(self.pending),
                "pending_count": self.pending_count, "gate_open": self.gate_open, "probes": self.probes}

    def load_state_dict(self, state):
        self.ema = None if state["ema"] is None else list(state["ema"])
        self.active = tuple(state["active"])
        self.pending = None if state["pending"] is None else tuple(state["pending"])
        self.pending_count = state["pending_count"]
        self.gate_open, self.probes = state["gate_open"], state["probes"]

    def select(self, energies, costs, core_energy):
        c = self.config
        sample = energy_shares(energies, core_energy, costs)
        self.probes += 1
        old_active = self.active
        if not any(sample[1:]) or not c.max_active_modes:
            # No factor signal: never keep stale active modes or EMA mass alive.
            self.ema = sample
            self.active, self.pending, self.pending_count, self.gate_open = (), None, 0, False
            return [], math.fsum(sample[1:]), self._diagnostics(sample, old_active, False, "no_factor_signal")
        self.ema = sample if self.ema is None else [
            c.score_ema * old + (1 - c.score_ema) * new for old, new in zip(self.ema, sample)]
        rho = math.fsum(self.ema[1:])
        low = max(0., c.escape_threshold - c.escape_hysteresis)
        high = min(1., c.escape_threshold + c.escape_hysteresis)
        # threshold=0 explicitly disables the family gate, including hysteresis.
        self.gate_open = c.escape_threshold == 0 or (
            rho >= (low if self.gate_open else high))
        proposal, feasible = coverage_subset(
            self.ema[1:], costs, c.max_active_modes, c.capture_fraction,
            c.cost_power, c.decision_tolerance) if self.gate_open else ((), False)
        if proposal and self.active:
            fractions = [v / rho for v in self.ema[1:]]
            coverage = math.fsum(fractions[i] for i in self.active)
            wanted = math.fsum(fractions[i] for i in proposal)
            max_cost = max(costs)
            price = lambda s: math.fsum((costs[i]/max_cost) ** c.cost_power for i in s)
            acceptable = (coverage >= c.capture_fraction - c.coverage_hysteresis if feasible
                          else coverage >= wanted - c.coverage_hysteresis)
            if (all(costs[i] > 0 for i in self.active) and len(self.active) <= c.max_active_modes
                    and acceptable and price(self.active) <= price(proposal) + c.decision_tolerance):
                proposal = self.active
        if proposal == self.active:
            self.pending, self.pending_count = None, 0
        else:
            self.pending_count = self.pending_count + 1 if proposal == self.pending else 1
            self.pending = proposal
            if self.pending_count >= c.confirm_probes:
                self.active, self.pending, self.pending_count = proposal, None, 0
        return list(self.active), math.fsum(sample[1:]), self._diagnostics(sample, old_active, feasible, "probe")

    def _diagnostics(self, sample, old_active, feasible, reason):
        factor_mass = math.fsum(self.ema[1:])
        return {
            "sample_energy_shares": sample, "smoothed_energy_shares": list(self.ema),
            "smoothed_escape_ratio": factor_mass, "gate_open": self.gate_open,
            "coverage_target_feasible": feasible, "selection_changed": self.active != old_active,
            "pending_modes": None if self.pending is None else list(self.pending),
            "pending_probes": self.pending_count, "selection_reason": reason,
            "smoothed_captured_fraction": (
                math.fsum(self.ema[i+1] for i in self.active)/factor_mass if factor_mass else 0.),
        }
