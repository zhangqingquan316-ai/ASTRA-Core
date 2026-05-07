import unittest

from astra_core.sweep import build_default_experiment_name, normalize_sweep_spec, sanitize_experiment_name
from astra_core.tasks import infer_families_from_modules


class PublicHelpersTest(unittest.TestCase):
    def test_sanitize_experiment_name(self):
        self.assertEqual(sanitize_experiment_name("astra core/sst2"), "astra_core_sst2")

    def test_normalize_sweep_spec(self):
        spec = normalize_sweep_spec(
            {
                "attn-ranks": [4, 4, 16, 32],
                "learning-rate": 1e-3,
            }
        )
        self.assertEqual(spec["attn_ranks"], [4, 4, 16, 32])
        self.assertEqual(spec["learning_rate"], 1e-3)

    def test_build_default_experiment_name(self):
        name = build_default_experiment_name(
            {
                "tuning_mode": "both",
                "attn_ranks": [4, 4, 16, 32],
                "attn_alpha": 1.0,
                "multiplicative_num_bases": 50,
                "num_train_epochs": 3,
            },
            index=2,
        )
        self.assertEqual(name, "exp_02_both_r4-4-16-32_a1.0_mb50_e3")

    def test_infer_families_from_modules(self):
        self.assertEqual(
            infer_families_from_modules(["query", "key", "value", "attention.output.dense"]),
            ["q", "k", "v", "o"],
        )


if __name__ == "__main__":
    unittest.main()
