import copy
import tempfile
import unittest

import tensorly as tl
import torch
from torch import nn

from astra_core.manifold import (ManifoldConfig, TangentAdamW, horizontal_direction,
                                 polar_retraction, select_modes)
from astra_core.modeling import TuckerFamilySharedState, TuckerAdaptedLinear
from astra_core.training_state import get_trainable_state, load_trainable_state


def orthogonal(n, r, dtype=torch.float64):
    return torch.linalg.qr(torch.randn(n, r, dtype=dtype), mode="reduced")[0]


def make_model(policy="manifold", dtype=torch.float64):
    torch.manual_seed(9)
    family = TuckerFamilySharedState("q", torch.randn(2, 2, 2, dtype=dtype),
                                    [orthogonal(3, 2, dtype)],
                                    [orthogonal(4, 2, dtype), orthogonal(5, 2, dtype)],
                                    policy, 3, 42)
    model = nn.Module()
    model.add_module("astra_core_shared_state_q", family)
    return model, family


def compose(family):
    factors = [family._factor_for_forward(p, i) for i, p in enumerate(family.manifold_factors())]
    return tl.tenalg.multi_mode_dot(family.get_updated_core(family.delta_core.dtype), factors)


class ManifoldGeometryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)

    def test_horizontal_descent_retraction_and_gauge_equivariance(self):
        u, g = orthogonal(7, 3), torch.randn(7, 3, dtype=torch.float64)
        core = torch.randn(3, 2, 4, dtype=torch.float64)
        d, energy = horizontal_direction(u, g, core, 0, 1e-3)
        torch.testing.assert_close(u.T @ d, torch.zeros(3, 3, dtype=u.dtype), atol=1e-12, rtol=0)
        self.assertGreater(energy, 0)
        torch.testing.assert_close((g * d).sum(), torch.tensor(energy, dtype=u.dtype))
        new = polar_retraction(u, -0.01 * d)
        torch.testing.assert_close(new.T @ new, torch.eye(3, dtype=u.dtype))
        q = orthogonal(3, 3)
        rotated_core = tl.tenalg.mode_dot(core, q.T, 0)
        rotated_d, rotated_energy = horizontal_direction(u @ q, g @ q, rotated_core, 0, 1e-3)
        torch.testing.assert_close(rotated_d, d @ q)
        self.assertAlmostEqual(energy, rotated_energy)
        torch.testing.assert_close(polar_retraction(u @ q, -0.01 * rotated_d), new @ q)

    def test_pairing_matches_retracted_directional_derivative(self):
        _, family = make_model()
        target = torch.randn(3, 4, 5, dtype=torch.float64)
        loss = ((compose(family) - target) ** 2).sum() / 2
        loss.backward()
        p = family.manifold_factors()[1]
        d, energy = horizontal_direction(p, p.grad, family.get_updated_core(p.dtype), 1, 1e-4)
        old = p.detach().clone()
        eps = 1e-6
        with torch.no_grad():
            p.copy_(polar_retraction(old, -eps * d))
            shifted = ((compose(family) - target) ** 2).sum() / 2
            p.copy_(old)
        self.assertAlmostEqual(float((shifted - loss.detach()) / eps), -energy, delta=1e-3)

    def test_degenerate_core_and_square_factor(self):
        u = orthogonal(5, 2)
        d, e = horizontal_direction(u, torch.randn_like(u), torch.zeros(2, 2, dtype=u.dtype), 0, 1e-3)
        self.assertTrue(torch.isfinite(d).all())
        self.assertGreaterEqual(e, 0)
        square = orthogonal(2, 2)
        d, e = horizontal_direction(square, torch.randn_like(square), torch.ones(2, 2), 0, 1e-3)
        self.assertEqual(e, 0)
        self.assertEqual(float(d.abs().sum()), 0)

    def test_selector_core_only_cost_and_budget(self):
        c = ManifoldConfig(max_active_modes=1)
        self.assertEqual(select_modes([1, 1], [3, 3], 100, c)[0], [])
        self.assertEqual(select_modes([10, 8, 100], [100, 2, 0], 1, c)[0], [1])
        self.assertEqual(select_modes([0, 0], [2, 3], 0, c)[0], [])

    def test_config_validation(self):
        for kwargs in ({"probe_interval": 0}, {"damping": 0}, {"escape_threshold": 2},
                       {"capture_fraction": 0}, {"cost_power": float("nan")}, {"warmup_steps": 0.5}):
            with self.assertRaises(ValueError):
                ManifoldConfig(**kwargs)

    def test_initial_forward_and_checkpoint_restore(self):
        for mode in ("additive", "multiplicative", "both", "manifold"):
            model, family = make_model(mode, torch.float32)
            base = nn.Linear(5, 4)
            x = torch.randn(2, 5)
            expected = base(x).detach()
            layer = TuckerAdaptedLinear(base, "q", family, (1,), (4, 5), "direct", 1.0)
            torch.testing.assert_close(layer(x), expected)
            if mode == "manifold":
                state = get_trainable_state(model)
                self.assertEqual(len(state), 4)
                with torch.no_grad():
                    family.delta_core.add_(1)
                    family.manifold_factors()[0].zero_()
                load_trainable_state(model, state)
                torch.testing.assert_close(layer(x), expected)

    def test_optimizer_schedule_freezing_and_resume(self):
        model, family = make_model()
        config = ManifoldConfig(warmup_steps=1, probe_interval=3, max_active_modes=1, escape_threshold=0)
        opt = TangentAdamW(model, lr=0.03, weight_decay=0.2, config=config)
        target = torch.randn(3, 4, 5, dtype=torch.float64)
        original = [p.detach().clone() for p in family.manifold_factors()]

        def train_step(m, f, optimizer):
            optimizer.zero_grad(set_to_none=True)
            ((compose(f) - target) ** 2).mean().backward()
            optimizer.step()

        train_step(model, family, opt)
        for p, old in zip(family.manifold_factors(), original):
            torch.testing.assert_close(p, old, atol=0, rtol=0)
        self.assertTrue(family.probe_factors)
        train_step(model, family, opt)
        self.assertEqual(len(family.active_modes), 1)
        self.assertFalse(family.probe_factors)
        opt.zero_grad(set_to_none=True)
        ((compose(family) - target) ** 2).mean().backward()
        for i, p in enumerate(family.manifold_factors()):
            self.assertEqual(p.grad is not None, i in family.active_modes)
            self.assertNotIn(p, opt.state)

        model2, family2 = make_model()
        model2.load_state_dict(model.state_dict())
        opt2 = TangentAdamW(model2)
        opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
        train_step(model, family, opt)
        train_step(model2, family2, opt2)
        for p, q in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)
        for p in family.manifold_factors():
            torch.testing.assert_close(p.T @ p, torch.eye(p.shape[1], dtype=p.dtype))

    def test_nonfinite_step_is_skipped(self):
        model, family = make_model()
        opt = TangentAdamW(model)
        before = get_trainable_state(model)
        family.delta_core.grad = torch.full_like(family.delta_core, float("nan"))
        opt.step()
        self.assertEqual(opt.completed_steps, 0)
        self.assertEqual(opt.skipped_steps, 1)
        for name, value in get_trainable_state(model).items():
            torch.testing.assert_close(value, before[name])

    def test_encoder_and_classifier_use_separate_adam_steps(self):
        model, family = make_model()
        model.add_module("classifier", nn.Sequential(nn.Linear(5, 4), nn.Linear(4, 2)).double())
        model.register_parameter("frozen", nn.Parameter(torch.ones(1), requires_grad=False))
        opt = TangentAdamW(model, lr=.02, classifier_lr=.0005, weight_decay=0,
                          config=ManifoldConfig(policy="core"))
        grouped = [id(p) for group in opt.param_groups for p in group["params"]]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual({id(p) for p in opt.param_groups[2]["params"]},
                         {id(p) for p in model.classifier.parameters()})
        before = get_trainable_state(model)
        for group in (opt.param_groups[0], opt.param_groups[2]):
            for p in group["params"]:
                p.grad = torch.ones_like(p)
        opt.step()
        for name, p in model.named_parameters():
            if name.startswith("classifier."):
                torch.testing.assert_close(before[name] - p, torch.full_like(p, .0005), atol=1e-10, rtol=1e-6)
        torch.testing.assert_close(before["astra_core_shared_state_q.delta_core"] - family.delta_core,
                                   torch.full_like(family.delta_core, .02), atol=1e-10, rtol=1e-6)
        state = copy.deepcopy(opt.state_dict())
        restored = TangentAdamW(model, lr=.001, classifier_lr=.001)
        restored.load_state_dict(state)
        self.assertEqual([g["lr"] for g in restored.param_groups], [.02, .02, .0005])

    def test_invalid_classifier_rate_is_rejected(self):
        model, _ = make_model()
        for rate in (0, -.001, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                TangentAdamW(model, classifier_lr=rate)

    def test_cpu_autocast_keeps_zero_initial_adapter(self):
        model, family = make_model(dtype=torch.float32)
        base = nn.Linear(5, 4)
        x = torch.randn(3, 5)
        weight = base.weight.detach().clone()
        bias = base.bias.detach().clone()
        layer = TuckerAdaptedLinear(base, "q", family, (1,), (4, 5), "direct", 1.0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            # Match the wrapper's separate bias addition rather than fused linear.
            # The wrapper replaces the stored weight with W0 - X0, so compare to the original W0.
            expected = x @ weight.T + bias
            actual = layer(x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_tangent_energies_equal_ambient_projection_blocks(self):
        _, family = make_model()
        ambient = torch.randn_like(compose(family))
        (compose(family) * ambient).sum().backward()
        factors = family.manifold_factors()
        core = family.base_core
        blocks = [tl.tenalg.multi_mode_dot(family.delta_core.grad, factors)]
        energy = family.delta_core.grad.square().sum()
        for mode, factor in enumerate(factors):
            direction, e = horizontal_direction(factor, factor.grad, core, mode, 0.0)
            block_factors = list(factors)
            block_factors[mode] = direction
            block = tl.tenalg.multi_mode_dot(core, block_factors)
            torch.testing.assert_close(block.square().sum(), torch.tensor(e, dtype=core.dtype))
            for previous in blocks:
                self.assertAlmostEqual(float((previous * block).sum().detach()), 0, places=10)
            blocks.append(block)
            energy = energy + e
        torch.testing.assert_close(sum(blocks).square().sum(), energy)

    def test_target_outside_fixed_subspace_is_learnable(self):
        errors = {}
        for policy in ("core", "all", "adaptive", "random"):
            model, family = make_model()
            with torch.no_grad():
                factors = [p.clone() for p in family.manifold_factors()]
                u = factors[1]
                h = torch.randn_like(u)
                h = h - u @ (u.T @ h)
                factors[1] = polar_retraction(u, 0.35 * h)
                target = tl.tenalg.multi_mode_dot(family.base_core, factors)
            opt = TangentAdamW(model, lr=0.15, weight_decay=0,
                              config=ManifoldConfig(policy=policy, warmup_steps=0, probe_interval=2,
                                                    escape_threshold=0, max_active_modes=1))
            for _ in range(100):
                opt.zero_grad(set_to_none=True)
                ((compose(family) - target) ** 2).sum().backward()
                opt.step()
            errors[policy] = float(((compose(family) - target) ** 2).sum().detach())
        self.assertLess(errors["adaptive"], 0.1 * errors["core"])
        self.assertLess(errors["all"], 0.1 * errors["core"])


class TrainerIntegrationTest(unittest.TestCase):
    def test_tiny_bert_accumulation_and_all_tensor_families(self):
        from datasets import Dataset
        from transformers import BertConfig, BertForSequenceClassification, Trainer, TrainingArguments
        from astra_core.modeling import (build_attention_tucker_basis_cache, build_ffn_tucker_basis_cache,
                                        build_runtime_family_from_decomposition_cache, patch_model_with_tucker,
                                        unfreeze_classifier_head)
        from astra_core.checkpoint import load_adapter_checkpoint
        torch.manual_seed(0)
        torch.set_num_threads(1)
        model = BertForSequenceClassification(BertConfig(
            vocab_size=24, hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=12, num_labels=2, hidden_dropout_prob=0, attention_probs_dropout_prob=0))
        model.eval()
        base_state = copy.deepcopy(model.state_dict())
        inputs = torch.randint(0, 24, (2, 5))
        with torch.no_grad():
            expected = model(input_ids=inputs).logits
        for p in model.parameters():
            p.requires_grad_(False)
        basis = []
        for family in ("q", "k", "v", "o", "ffn"):
            cache = (build_ffn_tucker_basis_cache(model, (1, 1, 2, 2)) if family == "ffn" else
                     build_attention_tucker_basis_cache(model, family, 2, (1, 1, 2, 2)))
            runtime = build_runtime_family_from_decomposition_cache(cache, 1, "manifold", 3, 0)
            basis.append(runtime)
            patch_model_with_tucker(model, runtime)
        unfreeze_classifier_head(model)
        with torch.no_grad():
            torch.testing.assert_close(model(input_ids=inputs).logits, expected, atol=1e-7, rtol=1e-5)
        data = Dataset.from_dict({"input_ids": inputs.tolist() * 4, "labels": [0, 1] * 4})
        opt = TangentAdamW(model, config=ManifoldConfig(warmup_steps=0, probe_interval=2, escape_threshold=0))
        with tempfile.TemporaryDirectory() as output:
            args = TrainingArguments(output_dir=output, max_steps=3, per_device_train_batch_size=2,
                                     gradient_accumulation_steps=2, report_to="none", save_strategy="no",
                                     disable_tqdm=True, max_grad_norm=0, use_cpu=True)
            trainer = Trainer(model=model, args=args, train_dataset=data, optimizers=(opt, None))
            trainer.train()
            from pathlib import Path
            torch.save(basis, Path(output) / "adapter_basis.pt")
            torch.save(get_trainable_state(model), Path(output) / "trainable_state.pt")
            fresh = BertForSequenceClassification(model.config)
            fresh.load_state_dict(base_state)
            load_adapter_checkpoint(fresh, output)
            fresh.eval()
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(fresh(input_ids=inputs).logits, model(input_ids=inputs).logits,
                                           atol=0, rtol=0)
        self.assertEqual(opt.completed_steps, 3)
        self.assertTrue(any(record["active_modes"] for record in opt.history))
        for family in opt.families:
            for p in family.manifold_factors():
                torch.testing.assert_close(p.T @ p, torch.eye(p.shape[1]), atol=2e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
