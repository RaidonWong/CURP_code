import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PipelineStaticTests(unittest.TestCase):
    def test_all_curp_stage2_training_entries_use_accumulate(self):
        entries = [
            "stage2.py",
            "stage2_nopq.py",
            "stage2_meanpool.py",
            "stage2_meanpool_sum.py",
            "stage2_dualmlp.py",
            "stage2_roberta.py",
            "stage2_bge.py",
            "stage2_bge_dualmlp.py",
            "stage2_roberta_dualmlp.py",
        ]
        for name in entries:
            source = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("accelerator.accumulate", source, name)

    def test_canonical_scripts_forward_task(self):
        task_source = (ROOT / "run_task.sh").read_text(encoding="utf-8")
        all_source = (ROOT / "run_all.sh").read_text(encoding="utf-8")
        self.assertIn('--task', task_source)
        self.assertIn('run_task.sh', all_source)

    def test_variant_inference_uses_shared_entry_point(self):
        for name in (
            "infer_dualmlp.py",
            "infer_nopq.py",
            "infer_meanpool.py",
            "infer_bge.py",
            "infer_bge_dualmlp.py",
            "infer_roberta.py",
            "infer_roberta_dualmlp.py",
        ):
            source = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("from infer import build_parser, main", source, name)


if __name__ == "__main__":
    unittest.main()
