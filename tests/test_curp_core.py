import json
import os
import re
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from stage2 import (
    DYNAMIC_TOKEN,
    STABLE_TOKEN,
    MeanPoolStable,
    DynamicCrossAttention,
    MLPProjection,
    PQCodebookModel,
    QFormer,
    Stage2Dataset,
    _save_checkpoint,
    inject_placeholder_embeddings,
    left_pad_tokenized_batch,
    tokenize_generation_prompt,
    tokenize_training_example,
)


class FakeTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, STABLE_TOKEN: 1, DYNAMIC_TOKEN: 2, "<eos>": 3}
        self.pad_token_id = 0
        self.eos_token_id = 3
        self._next_id = 10

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, -1)

    def _encode(self, text):
        pieces = re.findall(r"<USR_STABLE>|<USR_DYN>|\S+", text)
        result = []
        for piece in pieces:
            if piece not in self.vocab:
                self.vocab[piece] = self._next_id
                self._next_id += 1
            result.append(self.vocab[piece])
        return result

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = [100]
        for message in messages:
            ids.extend(self._encode(message["content"]))
        if add_generation_prompt:
            ids.extend([101, 102])
        return ids

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": self._encode(text)}


class TinyLM(nn.Module):
    def __init__(self, vocab_size=256, hidden_size=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def get_input_embeddings(self):
        return self.embedding


class CurpCoreTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.stable_id = self.tokenizer.convert_tokens_to_ids(STABLE_TOKEN)
        self.dynamic_id = self.tokenizer.convert_tokens_to_ids(DYNAMIC_TOKEN)

    def test_training_labels_only_cover_answer_and_preserve_placeholders(self):
        prompt = f"{STABLE_TOKEN} {STABLE_TOKEN} {DYNAMIC_TOKEN} " + "long " * 100
        ids, labels, positions = tokenize_training_example(
            self.tokenizer,
            prompt,
            "gold answer",
            max_length=24,
            stable_id=self.stable_id,
            dynamic_id=self.dynamic_id,
            ablation="both",
            num_query_tokens=2,
        )
        self.assertEqual(len(ids), 24)
        self.assertEqual(len(positions), 3)
        supervised = labels[labels != -100].tolist()
        self.assertEqual(supervised[-1], self.tokenizer.eos_token_id)
        self.assertGreaterEqual(len(supervised), 2)
        self.assertTrue(torch.all(labels[: min(positions)] == -100))

    def test_generation_rejects_missing_placeholder(self):
        with self.assertRaisesRegex(ValueError, "placeholder mismatch"):
            tokenize_generation_prompt(
                self.tokenizer,
                f"{STABLE_TOKEN} only one stable token",
                max_length=32,
                stable_id=self.stable_id,
                dynamic_id=self.dynamic_id,
                ablation="both",
                num_query_tokens=2,
            )

    def test_injection_validates_count_and_keeps_gradient(self):
        lm = TinyLM()
        ids = torch.tensor([[7, 1, 2, 8], [9, 1, 2, 6]])
        user = torch.randn(2, 2, 6, requires_grad=True)
        output = inject_placeholder_embeddings(lm, ids, user, [[1, 2], [1, 2]])
        output.sum().backward()
        self.assertIsNotNone(user.grad)
        self.assertTrue(torch.all(user.grad == 1))
        with self.assertRaisesRegex(ValueError, "expected 2"):
            inject_placeholder_embeddings(lm, ids, user.detach(), [[1], [1, 2]])

    def test_left_padding_adjusts_placeholder_positions(self):
        ids, _, mask, positions = left_pad_tokenized_batch(
            [torch.tensor([1, 5]), torch.tensor([1, 6, 7])],
            None,
            [[0], [0]],
            pad_token_id=0,
            device=torch.device("cpu"),
        )
        self.assertEqual(ids.tolist(), [[0, 1, 5], [1, 6, 7]])
        self.assertEqual(mask.tolist(), [[0, 1, 1], [1, 1, 1]])
        self.assertEqual(positions, [[1], [0]])

    def test_dataset_skips_invalid_orders_and_uses_safe_empty_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            embeddings_path = os.path.join(directory, "emb.pt")
            data_path = os.path.join(directory, "data.jsonl")
            embeddings = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            torch.save(embeddings, embeddings_path)
            rows = [
                {"input": "x", "output": "y", "profile": [{"order": 99}, {"order": 1}]},
                {"input": "a", "output": "b", "profile": [{"order": -1}]},
            ]
            with open(data_path, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            dataset = Stage2Dataset(
                data_path, embeddings_path, self.tokenizer, emb_dim=4
            )
            self.assertTrue(torch.equal(dataset[0]["profile_embeddings"], embeddings[1:2]))
            self.assertEqual(tuple(dataset[1]["profile_embeddings"].shape), (1, 4))
            self.assertEqual(float(dataset[1]["profile_embeddings"].abs().sum()), 0.0)

    def test_quantizer_variants_have_consistent_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = torch.randn(2, 3, 4)
            checkpoints = {
                "pq": {
                    "quant_type": "pq",
                    "codebooks": [torch.randn(5, 2), torch.randn(5, 2)],
                    "num_subspaces": 2,
                    "subspace_dim": 2,
                },
                "opq": {
                    "quant_type": "opq",
                    "codebooks": [torch.randn(5, 2), torch.randn(5, 2)],
                    "num_subspaces": 2,
                    "subspace_dim": 2,
                    "rotation": torch.eye(4),
                },
                "rq": {
                    "quant_type": "rq",
                    "codebooks": [torch.randn(5, 4), torch.randn(5, 4)],
                },
            }
            for quant_type, checkpoint in checkpoints.items():
                path = os.path.join(directory, f"{quant_type}.pt")
                torch.save(checkpoint, path)
                output, indices = PQCodebookModel(path)(inputs)
                self.assertEqual(tuple(output.shape), (2, 3, 4))
                self.assertEqual(tuple(indices.shape), (2, 3, 2))
                self.assertTrue(torch.isfinite(output).all())

    def test_mean_pool_ignores_padding(self):
        pool = MeanPoolStable(2)
        profile = torch.tensor([[[1.0, 3.0], [5.0, 7.0], [99.0, 99.0]]])
        result = pool(profile, torch.tensor([[False, False, True]]))
        self.assertTrue(torch.equal(result, torch.tensor([[[3.0, 5.0]]])))

    def test_finetuned_rq_checkpoint_round_trip_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            codebook_path = os.path.join(directory, "rq.pt")
            torch.save(
                {
                    "quant_type": "rq",
                    "codebooks": [torch.randn(5, 4), torch.randn(5, 4)],
                },
                codebook_path,
            )
            quantizer = PQCodebookModel(codebook_path)
            for parameter in quantizer.parameters():
                parameter.requires_grad = True
            qformer = QFormer(2, 4, 2, 1)
            dca = DynamicCrossAttention(4, 2)
            mlp = MLPProjection(4, 5, 6)
            optimizer = torch.optim.AdamW(
                list(qformer.parameters()) + list(dca.parameters()) + list(mlp.parameters())
            )
            args = SimpleNamespace(
                output_dir=directory,
                num_query_tokens=2,
                qformer_layers=1,
                num_heads=2,
                task="lamp4",
                ablation="both",
                max_length=128,
                max_profile_len=10,
                llm_path="llm",
                encoder_path="encoder",
                codebook_path=codebook_path,
                gradient_accumulation_steps=4,
                finetune_codebook=True,
            )
            _save_checkpoint(
                None,
                qformer,
                dca,
                mlp,
                args,
                step=7,
                pq_codebook_model=quantizer,
                optimizer=optimizer,
            )
            checkpoint = torch.load(
                os.path.join(directory, "stage2_checkpoint-7", "model.pt"),
                map_location="cpu",
            )
            self.assertEqual(checkpoint["format_version"], 3)
            self.assertEqual(checkpoint["quant_type"], "rq")
            self.assertEqual(checkpoint["num_stages"], 2)
            self.assertEqual(checkpoint["task"], "lamp4")
            self.assertIn("optimizer", checkpoint)
            self.assertEqual(checkpoint["llm_name"], "llm")
            self.assertEqual(checkpoint["encoder_name"], "encoder")
            self.assertEqual(checkpoint["codebook_name"], "rq.pt")
            self.assertNotIn("llm_path", checkpoint)
            self.assertNotIn("encoder_path", checkpoint)
            self.assertNotIn("codebook_path", checkpoint)


if __name__ == "__main__":
    unittest.main()
