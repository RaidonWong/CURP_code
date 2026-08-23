import unittest

from evaluate import (
    _parse_rating,
    compute_bleu,
    compute_classification,
    compute_mae_rmse,
    compute_rouge,
)


class EvaluationTests(unittest.TestCase):
    def test_lamp_labels_are_parsed_without_accepting_ambiguous_output(self):
        metrics = compute_classification(
            ["Answer: [1].", "[2]"],
            ["[1]", "2"],
            "lamp1",
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        lamp2 = compute_classification(
            ["The answer is dark comedy."], ["dark comedy"], "lamp2"
        )
        self.assertEqual(lamp2["accuracy"], 1.0)

    def test_rating_parser_rejects_out_of_range_or_multiple_numbers(self):
        self.assertEqual(_parse_rating("rating: 4"), 4.0)
        self.assertIsNone(_parse_rating("1 or 5"))
        self.assertIsNone(_parse_rating("9"))
        metrics = compute_mae_rmse(["rating: 4"], ["5"])
        self.assertEqual(metrics["mae"], 1.0)

    def test_generation_metrics_work_without_optional_nltk_or_rouge_packages(self):
        rouge = compute_rouge(["a short title"], ["a title"])
        bleu = compute_bleu(["a short title"], ["a title"])
        self.assertGreater(rouge["rouge1"], 0.0)
        self.assertGreaterEqual(bleu["bleu"], 0.0)


if __name__ == "__main__":
    unittest.main()
