import os
import tempfile
import unittest

import torch
from accelerate import Accelerator

from stage1 import CodebookDataset, CodebookModel, CodebookTrainer, balanced_kmeans_init


class Stage1Tests(unittest.TestCase):
    def test_dataset_keeps_remainder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "emb.pt")
            torch.save(torch.randn(10, 8), path)
            dataset = CodebookDataset(path, batch_size_per_sample=4)
            self.assertEqual(len(dataset), 3)
            self.assertEqual(tuple(dataset[2]["embeddings"].shape), (4, 8))

    def test_usage_regularizer_has_gradient(self):
        model = CodebookModel(codebook_size=7, emb_dim=8, num_subspaces=2)
        trainer = CodebookTrainer(model, Accelerator(cpu=True), usage_weight=1.0)
        total, _, _, usage, _ = trainer.compute_loss(
            {"embeddings": torch.randn(3, 5, 8)}
        )
        self.assertTrue(usage.requires_grad)
        total.backward()
        gradient = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)

    def test_balanced_kmeans_validates_sample_count(self):
        with self.assertRaisesRegex(ValueError, "at least as many samples"):
            balanced_kmeans_init(torch.randn(3, 4), 5, max_iters=1, device="cpu")


if __name__ == "__main__":
    unittest.main()
