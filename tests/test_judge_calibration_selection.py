import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "python" / "prepare_judge_calibration.py"
spec = importlib.util.spec_from_file_location("selection", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CalibrationSelectionTests(unittest.TestCase):
    def fixture(self):
        upstream = [{"question_id": str(i), "hypothesis": f"answer {i}",
                     "answer_score": int(i % 2 == 0), "question": "question",
                     "expected_answer": "reference", "judge_rubric": {},
                     "evaluation_type": kind}
                    for i, kind in enumerate(module.QUESTION_TYPES)]
        predictions = [{"qid": r["question_id"], "answer": r["hypothesis"]} for r in upstream]
        primary = [{"qid": r["question_id"], "status": "judged", "correct": bool(r["answer_score"])} for r in upstream]
        primary[0]["correct"] = not primary[0]["correct"]
        primary.append({"qid": "write-failure", "status": "service_error"})
        return primary, predictions, upstream

    def test_blinding_disagreements_and_failed_denominator(self):
        p, a, u = self.fixture()
        blind, comparisons, review, summary = module.prepare(p, a, u)
        self.assertEqual(summary["selection_counts"], {"disagreement": 1, "agreement": 5})
        self.assertEqual(summary["primary_without_saved_answer"], 1)
        self.assertEqual(summary["review_completed"], 0)
        self.assertTrue(all(r["correct"] is None for r in review))
        self.assertTrue(all("primary_correct" not in r and "selection" not in r for r in blind))
        self.assertEqual(module.prepare(p, a, list(reversed(u))), (blind, comparisons, review, summary))

    def test_different_answers_and_duplicates_are_rejected(self):
        p, a, u = self.fixture()
        a[0]["answer"] = "changed answer"
        with self.assertRaisesRegex(ValueError, "same saved answer"):
            module.prepare(p, a, u)
        p, a, u = self.fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            module.prepare(p, a, u + [u[0]])


if __name__ == "__main__":
    unittest.main()
