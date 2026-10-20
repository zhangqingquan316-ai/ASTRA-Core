"""Numerical and controller properties, independent of GLUE scores."""
import copy
import unittest

import torch
import tensorly as tl
from torch import nn

from astra_core.modeling import TuckerAdaptedLinear, TuckerFamilySharedState
from astra_core.manifold import ManifoldConfig, TangentAdamW, polar_retraction
from astra_core.stability import (StableModeSelector, energy_shares, coverage_subset,
                                  normalized_tangent_statistics, incremental_polar,
                                  capped_factor_step)
from test_manifold import make_model, orthogonal, compose


class StableAnchorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def layer(self, mode="manifold", dtype=torch.float64, layout="delta"):
        model, family = make_model(mode, dtype)
        base = nn.Linear(5, 4).to(dtype)
        layer = TuckerAdaptedLinear(base, "q", family, (1,), (4, 5), "direct", .7, layout)
        return model, family, layer

    def test_delta_zero_is_exact_original_linear_with_amp(self):
        for mode in ("manifold", "additive", "multiplicative", "both"):
            for amp in (False, True):
                _, _, layer = self.layer(mode, torch.float32)
                x = torch.randn(3, 5)
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
                    expected, actual = layer.base_layer(x), layer(x)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(torch.equal(actual, expected))

    def test_delta_algebra_and_all_gradients_match_float64_reference(self):
        for mode in ("manifold", "additive", "multiplicative", "both"):
            _, family, layer = self.layer(mode)
            with torch.no_grad():
                if family.has_additive:
                    family.delta_core.normal_(std=.01)
                if family.is_manifold:
                    for p in family.manifold_factors():
                        p.add_(torch.randn_like(p) * .02)
                for adapter in family.mode_transform_adapters:
                    adapter.coefficients.normal_(std=.03)
            x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
            current = family.apply_mode_transforms(family.get_updated_core(x.dtype), x.dtype)
            current = tl.tenalg.multi_mode_dot(current, family.manifold_factors())[1]
            initial = tl.tenalg.multi_mode_dot(family.base_core, [
                getattr(family, f"anchor_factor_{i}") for i in range(3)])[1]
            reference = layer.base_layer(x) + .7 * (x @ (current-initial).T)
            actual = layer(x)
            torch.testing.assert_close(actual, reference, atol=1e-14, rtol=1e-12)
            params = [x] + [p for p in family.parameters() if p.requires_grad]
            ga = torch.autograd.grad(actual.square().sum(), params, retain_graph=True)
            gb = torch.autograd.grad(reference.square().sum(), params)
            for a, b in zip(ga, gb):
                torch.testing.assert_close(a, b, atol=1e-13, rtol=1e-11)

    def test_sub_ulp_core_update_survives_reconstruction(self):
        _, family, layer = self.layer(dtype=torch.float32)
        with torch.no_grad():
            family.delta_core.fill_(1e-12)
        delta = layer._compose_delta_tensor(torch.float32)
        self.assertGreater(float(delta.abs().max()), 1e-14)
        expected = tl.tenalg.multi_mode_dot(family.delta_core.double(), [
            p.double() for p in family.manifold_factors()])[1]
        torch.testing.assert_close(delta.double(), expected, atol=1e-19, rtol=2e-6)
        # The discarded alternative really does lose this increment.
        self.assertTrue(torch.equal(family.base_core + family.delta_core, family.base_core))

    def test_two_contracted_modes_transpose_and_inactive_factor_history(self):
        core = torch.randn(2, 2, 2, 2, dtype=torch.float64)
        factors = [orthogonal(n, 2) for n in (3, 4, 5, 6)]
        family = TuckerFamilySharedState("ffn", core, factors[:2], factors[2:], "manifold", 3, 42)
        layer = TuckerAdaptedLinear(nn.Linear(5, 6).double(), "ffn", family,
                                   (1, 2), (6, 5), "transpose", 1., "delta")
        with torch.no_grad():
            family.delta_core.normal_(std=.01)
            family.manifold_factors()[1].add_(.01)
        family.probe_factors = False
        family.active_modes = {2}
        current = tl.tenalg.multi_mode_dot(family.base_core + family.delta_core,
                                           family.manifold_factors())[1, 2]
        initial = tl.tenalg.multi_mode_dot(core, factors)[1, 2]
        x = torch.randn(3, 5, dtype=torch.float64)
        expected = layer.base_layer(x) + x @ (current-initial)
        torch.testing.assert_close(layer(x), expected, atol=1e-14, rtol=1e-12)
        layer(x).sum().backward()
        for i, p in enumerate(family.manifold_factors()):
            self.assertEqual(p.grad is not None, i == 2)

    def test_cpu_anchor_matches_two_matrix_outputs_and_gradients(self):
        m, f, layer = self.layer(dtype=torch.float32, layout="two_matrix")
        m2, f2, layer2 = self.layer(dtype=torch.float32, layout="cpu_anchor")
        with torch.no_grad():
            f.delta_core.normal_(std=.1)
            f2.delta_core.copy_(f.delta_core)
        x = torch.randn(3, 5)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y, y2 = layer(x), layer2(x)
        self.assertTrue(torch.equal(y, y2))
        y.sum().backward()
        y2.sum().backward()
        for p, q in zip(m.parameters(), m2.parameters()):
            self.assertTrue(torch.equal(p.grad, q.grad))
        self.assertNotIn("cpu_anchor", dict(layer2.named_buffers()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA regression")
    def test_cuda_anchor_zero_and_tiny_increment(self):
        _, family, layer = self.layer(dtype=torch.float32)
        family.cuda()
        layer.cuda()
        x = torch.randn(3, 5, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.assertTrue(torch.equal(layer(x), layer.base_layer(x)))
        with torch.no_grad():
            family.delta_core.fill_(1e-12)
        self.assertGreater(float(layer._compose_delta_tensor(torch.float32).abs().max()), 1e-14)
        y = layer(x).sum()
        y.backward()
        self.assertTrue(torch.isfinite(family.delta_core.grad).all())

    def test_delta_checkpoint_metadata_restores_identical_logits(self):
        import json
        import tempfile
        from pathlib import Path
        from transformers import BertConfig, BertForSequenceClassification
        from astra_core.modeling import (build_attention_tucker_basis_cache,
                                        build_runtime_family_from_decomposition_cache,
                                        patch_model_with_tucker, unfreeze_classifier_head)
        from astra_core.checkpoint import load_adapter_checkpoint
        from astra_core.training_state import get_trainable_state
        config = BertConfig(vocab_size=24, hidden_size=8, num_hidden_layers=2,
                            num_attention_heads=2, intermediate_size=12, num_labels=2)
        model = BertForSequenceClassification(config)
        initial = copy.deepcopy(model.state_dict())
        for p in model.parameters():
            p.requires_grad_(False)
        basis = build_runtime_family_from_decomposition_cache(
            build_attention_tucker_basis_cache(model, "q", 2, (1, 1, 2, 2)), 1, "manifold", 3, 42)
        patch_model_with_tucker(model, basis, weight_layout="delta")
        unfreeze_classifier_head(model)
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p)*.01)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save([basis], root/"adapter_basis.pt")
            torch.save(get_trainable_state(model), root/"trainable_state.pt")
            (root/"training_config.json").write_text(json.dumps({"weight_layout": "delta"}))
            fresh = BertForSequenceClassification(config)
            fresh.load_state_dict(initial)
            load_adapter_checkpoint(fresh, directory)
        model.eval()
        fresh.eval()
        inputs = torch.randint(0, 24, (2, 5))
        with torch.no_grad():
            self.assertTrue(torch.equal(model(inputs).logits, fresh(inputs).logits))


class StableGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_zero_and_tiny_steps_do_not_inject_retraction_noise(self):
        u = orthogonal(32, 8, torch.float32)
        self.assertTrue(torch.equal(incremental_polar(u, torch.zeros_like(u)), u))
        self.assertTrue(torch.equal(incremental_polar(u, torch.ones_like(u) * 1e-12), u))

    def test_incremental_polar_matches_reference_and_bounds_drift(self):
        u = orthogonal(17, 4)
        s = torch.randn_like(u) * .01
        torch.testing.assert_close(incremental_polar(u, s), polar_retraction(u, s), atol=1e-14, rtol=1e-12)
        u = u.float()
        for _ in range(500):
            g = torch.randn_like(u).double()
            h = g - u.double() @ (u.double().T @ g)
            u = incremental_polar(u, h * 1e-4)
        self.assertLess(float(torch.linalg.matrix_norm(u.T @ u - torch.eye(4))), 1e-5)

    def test_gradient_scale_invariance_before_squaring(self):
        _, f = make_model()
        factors = f.manifold_factors()
        grads = [torch.randn_like(p) for p in factors]
        gc = torch.randn_like(f.base_core)
        reference = None
        for scale in (1., 1e-150, 1e150):
            for p, g in zip(factors, grads):
                p.grad = g * scale
            directions, energies, core, _ = normalized_tangent_statistics(
                factors, f.base_core, gc*scale, .001)
            shares = energy_shares(energies, core, [2, 4, 6])
            if reference is None:
                reference = shares, directions
            torch.testing.assert_close(torch.tensor(shares), torch.tensor(reference[0]), atol=1e-7, rtol=1e-6)
            for i, d in directions.items():
                torch.testing.assert_close(d/scale, reference[1][i], atol=1e-13, rtol=1e-12)

    def test_extreme_direction_cap_and_zero_lr(self):
        for scale in (1e-300, 1e300):
            d = torch.full((10, 3), scale, dtype=torch.float64)
            step = capped_factor_step(d, .001, .05)
            self.assertTrue(torch.isfinite(step).all())
            self.assertLessEqual(float(torch.linalg.vector_norm(step)), .05000000001)
            self.assertEqual(torch.count_nonzero(capped_factor_step(d, 0, .05)), 0)

    def test_float32_gradient_quadratics_do_not_underflow_or_overflow(self):
        _, family = make_model(dtype=torch.float32)
        factors = family.manifold_factors()
        gradients = [torch.randn_like(p) for p in factors]
        core_gradient = torch.randn_like(family.base_core)
        shares = []
        for scale in (1., 1e-30, 1e30):
            for factor, gradient in zip(factors, gradients):
                factor.grad = gradient * scale
            _, energies, core, _ = normalized_tangent_statistics(
                factors, family.base_core, core_gradient*scale, .001)
            shares.append(torch.tensor(energy_shares(energies, core, [2, 4, 6])))
        for sample in shares[1:]:
            torch.testing.assert_close(sample, shares[0], atol=1e-7, rtol=1e-6)


class StableSelectionTests(unittest.TestCase):
    def config(self, **kwargs):
        return ManifoldConfig(policy="stable", numerical_mode="stable", **kwargs)

    def test_all_small_all_large_and_ties_have_identical_trajectories(self):
        trajectories = []
        for scale in (1., 1e-300, 1e300):
            s = StableModeSelector(self.config())
            trajectory = []
            for n in range(120):
                e = [(.5 if n < 60 else .2)*scale, .5*scale, .001*scale, 100*scale]
                modes, _, _ = s.select(e, [1, 1, 2, 0], .1*scale)
                trajectory.append(modes)
            trajectories.append(trajectory)
        self.assertEqual(trajectories[0], trajectories[1])
        self.assertEqual(trajectories[0], trajectories[2])

    def test_zero_signal_clears_history_and_pending(self):
        s = StableModeSelector(self.config())
        for _ in range(4):
            s.select([10, 1], [1, 1], .1)
        self.assertTrue(s.active)
        modes, _, record = s.select([0, 0], [1, 1], 0)
        self.assertEqual(modes, [])
        self.assertIsNone(s.pending)
        self.assertEqual(record["selection_reason"], "no_factor_signal")

    def test_exact_coverage_avoids_cheap_negligible_modes(self):
        modes, feasible = coverage_subset([.001, .999], [1, 1000], 1, .8, 1, .001)
        self.assertEqual(modes, (1,))
        self.assertTrue(feasible)
        modes, feasible = coverage_subset([.34, .33, .33], [1, 2, 3], 1, .8, 1, .001)
        self.assertEqual(modes, (0,))
        self.assertFalse(feasible)
        self.assertEqual(coverage_subset([1, 2], [0, 0], 2, .8, 1, .001)[0], ())

    def test_confirmation_and_near_tie_hysteresis(self):
        s = StableModeSelector(self.config(max_active_modes=1, score_ema=0., escape_threshold=0))
        self.assertEqual(s.select([1, 0], [1, 1], 0)[0], [])
        self.assertEqual(s.select([1, 0], [1, 1], 0)[0], [])
        self.assertEqual(s.select([1, 0], [1, 1], 0)[0], [0])
        for n in range(100):
            self.assertEqual(s.select([.5+(-1)**n*1e-5, .5], [1, 1], 0)[0], [0])
        for _ in range(3):
            modes = s.select([0, 1], [1, 1], 0)[0]
        self.assertEqual(modes, [1])

    def test_controller_and_optimizer_resume_exactly(self):
        model, f = make_model()
        opt = TangentAdamW(model, config=self.config(warmup_steps=0, probe_interval=1, escape_threshold=0))
        target = torch.randn_like(compose(f))
        def step(family, optimizer):
            optimizer.zero_grad(set_to_none=True)
            (compose(family)-target).square().sum().backward()
            optimizer.step()
        for _ in range(2):  # pending activation must survive checkpointing
            step(f, opt)
        m2, f2 = make_model()
        m2.load_state_dict(model.state_dict())
        o2 = TangentAdamW(m2)
        o2.load_state_dict(copy.deepcopy(opt.state_dict()))
        for _ in range(10):
            step(f, opt)
            step(f2, o2)
        for p, q in zip(model.parameters(), m2.parameters()):
            self.assertTrue(torch.equal(p, q))
        self.assertEqual(opt.history, o2.history)
        self.assertEqual(opt.selectors[0].state_dict(), o2.selectors[0].state_dict())

    def test_default_smoothing_rejects_spike_but_tracks_persistent_change(self):
        selector = StableModeSelector(self.config(max_active_modes=1, escape_threshold=0))
        for _ in range(50):
            selector.select([1., 0.], [1, 1], 0.)
        self.assertEqual(selector.select([0., 1.], [1, 1], 0.)[0], [0])
        for _ in range(5):
            self.assertEqual(selector.select([1., 0.], [1, 1], 0.)[0], [0])
        changed_at = None
        for probe in range(30):
            modes, _, _ = selector.select([0., 1.], [1, 1], 0.)
            if modes == [1] and changed_at is None:
                changed_at = probe + 1
        self.assertIsNotNone(changed_at)
        self.assertGreater(changed_at, 3)
        self.assertLessEqual(changed_at, 20)
        self.assertEqual(modes, [1])

    def test_escape_schmitt_gate_has_separate_open_and_close_thresholds(self):
        selector = StableModeSelector(self.config(score_ema=0., confirm_probes=1))
        for rho, expected in ((.17, False), (.19, True), (.14, True), (.11, False)):
            modes, _, diagnostics = selector.select([rho], [1], 1-rho)
            self.assertEqual(diagnostics["gate_open"], expected)
            self.assertEqual(bool(modes), expected)

    def test_invalid_configuration_and_energy_rejected(self):
        for kw in ({"score_ema": 1}, {"confirm_probes": 0}, {"decision_tolerance": -1},
                   {"coverage_hysteresis": float("nan")}, {"orthogonality_tolerance": 0}):
            with self.assertRaises(ValueError):
                self.config(**kw)
        for energy in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                energy_shares([energy], 1, [1])
