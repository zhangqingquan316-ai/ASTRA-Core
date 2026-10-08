"""Verify real LR warmup, both TangentAdamW groups, and MRPC metric recording."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from datasets import Dataset
from transformers import BertConfig, BertForSequenceClassification, Trainer, TrainerCallback

from astra_core.artifacts import extract_history_records, save_trainable_checkpoint
from astra_core.experiment import build_training_arguments, compute_parameter_statistics
from astra_core.manifold import ManifoldConfig, TangentAdamW
from astra_core.modeling import (build_attention_tucker_basis_cache,
                                 build_runtime_family_from_decomposition_cache,
                                 patch_model_with_tucker, unfreeze_classifier_head)
from astra_core.tasks import build_compute_metrics, compute_binary_f1
from astra_core.training_state import OptimizerScheduleCallback


def experiment_args(**updates):
    values = dict(model_path="test-model", glue_task="mrpc", target_modules=None,
                  attn_ranks=[1, 1, 2, 2], ffn_ranks=[1, 1, 2, 2], attn_alpha=1., ffn_alpha=1.,
                  tuning_mode="manifold", multiplicative_num_bases=3, max_length=5,
                  learning_rate=1e-3, per_device_train_batch_size=2, per_device_eval_batch_size=2,
                  num_train_epochs=3, weight_decay=.01, seed=42, gradient_accumulation_steps=1,
                  warmup_ratio=.5, warmup_steps=0, lr_scheduler_type="linear")
    values.update(updates)
    return SimpleNamespace(**values)


class OptimizerScheduleTest(unittest.TestCase):
    def test_invalid_or_ambiguous_warmup(self):
        for updates in ({"warmup_ratio": -.1}, {"warmup_ratio": float("nan")},
                        {"warmup_steps": -1}, {"warmup_steps": 1}, {"warmup_steps": .5}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                build_training_arguments(experiment_args(**updates), Path("unused"))

    def test_old_callers_keep_zero_warmup(self):
        args = experiment_args()
        del args.warmup_ratio, args.warmup_steps, args.lr_scheduler_type
        training = build_training_arguments(args, Path("unused"))
        self.assertEqual(training.get_warmup_steps(4020), 0)
        self.assertEqual(training.lr_scheduler_type.value, "linear")

    def test_mrpc_metrics_include_f1_and_handle_no_positives(self):
        labels = np.array([0, 1, 1, 0])
        logits = np.array([[2, 0], [0, 2], [0, 2], [0, 2]])
        metrics = build_compute_metrics("mrpc")((logits, labels))
        self.assertAlmostEqual(metrics["accuracy"], .75)
        self.assertAlmostEqual(metrics["f1"], .8)
        self.assertAlmostEqual(metrics["accuracy_f1"], .775)
        self.assertEqual(compute_binary_f1(np.zeros(4), np.zeros(4)), 0.)
        self.assertEqual(set(build_compute_metrics("sst2")((logits, labels))), {"accuracy"})

    def test_histories_keep_lr_and_mrpc_metrics(self):
        train, evaluation = extract_history_records([
            {"loss": .3, "epoch": 1, "learning_rate": 1e-4},
            {"eval_loss": .4, "epoch": 1, "eval_accuracy": .75,
             "eval_f1": .8, "eval_accuracy_f1": .775},
        ], "eval_accuracy")
        self.assertEqual(train[0]["learning_rate"], 1e-4)
        self.assertEqual(evaluation[0]["eval_f1"], .8)

    def test_trainer_warms_and_decays_both_tangent_groups(self):
        self._check_trainer_schedule(None)

    def test_trainer_preserves_distinct_encoder_and_classifier_schedules(self):
        self._check_trainer_schedule(5e-4)

    def _check_trainer_schedule(self, classifier_lr):
        torch.manual_seed(0)
        torch.set_num_threads(1)
        model = BertForSequenceClassification(BertConfig(
            vocab_size=24, hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=12, num_labels=2, hidden_dropout_prob=0, attention_probs_dropout_prob=0))
        for p in model.parameters():
            p.requires_grad_(False)
        cache = build_attention_tucker_basis_cache(model, "q", 2, (1, 1, 2, 2))
        basis = build_runtime_family_from_decomposition_cache(cache, 1., "manifold", 3, 0)
        patch_model_with_tucker(model, basis)
        unfreeze_classifier_head(model)
        encoder_lr = 2e-2 if classifier_lr is not None else 1e-3
        optimizer = TangentAdamW(model, lr=encoder_lr, classifier_lr=classifier_lr, config=ManifoldConfig(
            policy="all", warmup_steps=0, factor_lr_scale=.5))
        rows = {"input_ids": [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]] * 4, "labels": [0, 1] * 4}
        data = Dataset.from_dict(rows)
        rates = []

        class RateRecorder(TrainerCallback):
            def on_step_end(self, args, state, control, optimizer=None, **kwargs):
                rates.append([g["lr"] for g in optimizer.param_groups])

        with tempfile.TemporaryDirectory() as output:
            args = experiment_args(learning_rate=encoder_lr, classifier_learning_rate=classifier_lr)
            training = build_training_arguments(args, Path(output))
            training.max_steps = 6
            training.logging_steps = 1
            training.disable_tqdm = True
            trainer = Trainer(model=model, args=training, train_dataset=data, eval_dataset=data,
                              optimizers=(optimizer, None), callbacks=[OptimizerScheduleCallback(args), RateRecorder()])
            trainer.train()
            self.assertEqual(optimizer.completed_steps, 6)
            self.assertEqual(args._optimizer_schedule["actual_warmup_steps"], 3)
            self.assertEqual(args._optimizer_schedule["total_optimizer_steps"], 6)
            np.testing.assert_allclose(np.array(rates)[:, 0],
                                       np.array([1/3, 2/3, 1, 2/3, 1/3, 0]) * encoder_lr, atol=1e-12)
            np.testing.assert_allclose(np.array(rates)[:, 1], np.array(rates)[:, 0] * .5, atol=1e-12)
            if classifier_lr is not None:
                np.testing.assert_allclose(np.array(rates)[:, 2],
                                           np.array(rates)[:, 0] * classifier_lr / encoder_lr, atol=1e-12)
                scheduled = args._optimizer_schedule["parameter_groups"]
                self.assertEqual([g["name"] for g in scheduled],
                                 ["encoder_core", "encoder_factors", "classifier"])
                self.assertEqual(scheduled[2]["peak_learning_rate"], classifier_lr)
            save_trainable_checkpoint(model, Path(output) / "final", args, ["q"],
                                      "schedule-test", "test-data", compute_parameter_statistics(model))
            metadata = json.loads((Path(output) / "final/training_config.json").read_text())
            self.assertEqual(metadata["warmup_ratio"], .5)
            self.assertEqual(metadata["optimizer_schedule"]["actual_warmup_steps"], 3)
            self.assertEqual(metadata["classifier_learning_rate"], classifier_lr or encoder_lr)


if __name__ == "__main__":
    unittest.main()
