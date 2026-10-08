"""Exercise STS-B's float labels and one-output regression with split rates."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from datasets import Dataset
from transformers import BertConfig, BertForSequenceClassification, Trainer

from astra_core.experiment import build_training_arguments
from astra_core.manifold import ManifoldConfig, TangentAdamW
from astra_core.modeling import (build_attention_tucker_basis_cache,
                                 build_runtime_family_from_decomposition_cache,
                                 patch_model_with_tucker, unfreeze_classifier_head)
from astra_core.tasks import build_compute_metrics
from astra_core.training_state import OptimizerScheduleCallback


class RemainingGlueTest(unittest.TestCase):
    def test_stsb_regression_trains_encoder_and_single_output_head(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        model = BertForSequenceClassification(BertConfig(
            vocab_size=24, hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=12, num_labels=1, problem_type="regression",
            hidden_dropout_prob=0, attention_probs_dropout_prob=0))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        cache = build_attention_tucker_basis_cache(model, "q", 2, (1, 1, 2, 2))
        family = build_runtime_family_from_decomposition_cache(cache, 1., "manifold", 3, 0)
        patch_model_with_tucker(model, family)
        unfreeze_classifier_head(model)
        initial_head = model.classifier.weight.detach().clone()
        optimizer = TangentAdamW(model, lr=.002, classifier_lr=.0001, config=ManifoldConfig())
        data = Dataset.from_dict({
            "input_ids": [[1,2,3], [2,3,4], [3,4,5], [4,5,6]],
            "labels": [.5, 2., 4., 5.],
        })
        args = SimpleNamespace(tuning_mode="manifold", learning_rate=.002, classifier_learning_rate=.0001,
                               warmup_ratio=.06, warmup_steps=0, lr_scheduler_type="linear",
                               per_device_train_batch_size=2, per_device_eval_batch_size=2,
                               num_train_epochs=20, weight_decay=.01, seed=42,
                               gradient_accumulation_steps=1)
        metrics = build_compute_metrics("stsb")
        perfect = metrics((np.array([[.6], [2.1], [4.1], [5.1]]), np.array([.5, 2., 4., 5.])))
        self.assertAlmostEqual(perfect["pearson"], 1.)
        with tempfile.TemporaryDirectory() as output:
            training = build_training_arguments(args, Path(output))
            training.max_steps = 4
            training.disable_tqdm = True
            training.use_cpu = True
            trainer = Trainer(model=model, args=training, train_dataset=data, eval_dataset=data,
                              compute_metrics=metrics, optimizers=(optimizer, None),
                              callbacks=[OptimizerScheduleCallback(args)])
            trainer.train()
            result = trainer.predict(data)
            self.assertEqual(result.predictions.shape, (4, 1))
            self.assertTrue(np.issubdtype(result.label_ids.dtype, np.floating))
            self.assertTrue(np.isfinite(result.predictions).all())
            self.assertTrue(np.isfinite(result.metrics["test_loss"]))
            model.eval()
            with torch.no_grad():
                batch = dict(input_ids=torch.tensor([[1,2,3], [2,3,4]]), labels=torch.tensor([.5, 2.]))
                forward = model(**batch)
                torch.testing.assert_close(forward.loss, ((forward.logits.squeeze(-1) - batch["labels"]) ** 2).mean())
        self.assertEqual(optimizer.completed_steps, 4)
        self.assertGreater(int(torch.count_nonzero(model.astra_core_shared_state_q.delta_core)), 0)
        self.assertFalse(torch.allclose(model.classifier.weight, initial_head))
        self.assertEqual(args._optimizer_schedule["classifier_peak_learning_rate"], .0001)


if __name__ == "__main__":
    unittest.main()
