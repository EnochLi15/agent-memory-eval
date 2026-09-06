import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from merge_calibration_retry import merge


class MergeRetryTests(unittest.TestCase):
    def test_preserves_semantics_and_denominator(self):
        original = [{'qid': 'a', 'status': 'judged', 'correct': False}, {'qid': 'b', 'status': 'protocol_error', 'correct': None}, {'qid': 'c', 'status': 'error', 'correct': None}]
        result = merge(original, [{'qid': 'c', 'status': 'needs_review', 'correct': None}])
        self.assertEqual(len(result), 3)
        self.assertFalse(result[0]['correct'])
        self.assertEqual(result[1]['status'], 'protocol_error')
        self.assertEqual(result[2]['audit_origin'], 'transport_retry')
        self.assertEqual(original[2]['status'], 'error')

    def test_rejects_resampling_semantic_cases_missing_rows_and_duplicates(self):
        old = [{'qid': 'a', 'status': 'judged'}, {'qid': 'b', 'status': 'error'}]
        for retry in ([], [{'qid': 'a', 'status': 'judged'}], [{'qid': 'b', 'status': 'judged'}]*2):
            with self.assertRaises(ValueError):
                merge(old, retry)
