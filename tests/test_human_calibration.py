import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from report_human_calibration import report


class HumanCalibrationTests(unittest.TestCase):
    def fixture(self):
        cases = [{'qid': str(i), 'answer': 'The saved answer.', 'evaluation_type': 'type' + str(i % 2),
                  'rubric': {'must_include': ['required meaning'], 'must_not_include': ['old value']}}
                 for i in range(4)]
        prior = [{'qid': str(i), 'primary_correct': i < 2, 'upstream_correct': i % 2 == 0,
                  'selection': 'agreement' if (i < 2) == (i % 2 == 0) else 'disagreement'} for i in range(4)]
        reviews = [{'qid': str(i), 'reviewer': 'fixture-reviewer', 'reviewer_kind': 'human',
                    'correct': i % 2 == 0, 'criteria': [], 'reason': 'Synthetic unit-test rationale.', 'status': 'reviewed'} for i in range(4)]
        roster = [{'reviewer': 'fixture-reviewer', 'reviewer_kind': 'human', 'independent_of_model_judging': True,
                   'attestation': 'Synthetic test fixture only; not a real human judgment.'}]
        return cases, prior, reviews, roster

    def test_confusion_denominators_and_strata_keep_both_error_directions(self):
        cases, prior, reviews, roster = self.fixture()
        before = copy.deepcopy((cases, prior, reviews, roster))
        summary, accepted = report(cases, prior, reviews, roster)
        self.assertEqual(summary['status'], 'ready_for_calibration_review')
        self.assertEqual(summary['overall']['primary'], {'true_positive': 1, 'true_negative': 1, 'false_positive': 1, 'false_negative': 1, 'compared': 4, 'agreement': .5})
        self.assertEqual(summary['overall']['upstream']['agreement'], 1)
        self.assertEqual(summary['by_selection']['disagreement']['primary']['agreement'], 0)
        self.assertEqual(summary['by_question_type']['type0']['primary']['false_negative'], 1)
        self.assertEqual(len(accepted), 4)
        self.assertEqual((cases, prior, reviews, roster), before)

    def test_pending_missing_uncertain_never_become_negative_or_positive_labels(self):
        cases, prior, reviews, roster = self.fixture()
        reviews[0].update(status='pending_independent_review', reviewer=None, reviewer_kind=None, correct=None, reason=None)
        reviews[1].update(status='uncertain', correct=None)
        reviews.pop()
        summary, accepted = report(cases, prior, reviews, roster)
        self.assertEqual(summary['selected_cases'], 4)
        self.assertEqual(summary['compared_independent_labels'], 1)
        self.assertEqual(summary['review_status_counts'], {'pending_independent_review': 1, 'uncertain': 1, 'reviewed': 1, 'missing_review': 1})
        summary, accepted = report(cases, prior, [], roster)
        self.assertIsNone(summary['overall']['primary']['agreement'])
        self.assertEqual(accepted, [])

    def test_unattested_identity_is_not_an_independent_reference_label(self):
        cases, prior, reviews, roster = self.fixture()
        summary, _ = report(cases, prior, reviews)
        self.assertEqual(summary['validated_binary_human_labels'], 4)
        self.assertEqual(summary['compared_independent_labels'], 0)
        self.assertIsNone(summary['overall']['primary']['agreement'])
        self.assertEqual(summary['status'], 'incomplete_independent_review')
        for bad in [{'reviewer_kind': 'model'}, {'independent_of_model_judging': False}, {'attestation': ''}]:
            with self.assertRaisesRegex(ValueError, 'attestation'):
                report(cases, prior, reviews, [{**roster[0], **bad}])

    def test_nonhuman_integer_labels_blank_reasons_and_pending_verdicts_reject(self):
        cases, prior, reviews, roster = self.fixture()
        for patch in [{'reviewer_kind': 'model'}, {'correct': 1}, {'reason': ' '}, {'reviewer': None},
                      {'status': 'uncertain', 'correct': False}, {'status': 'pending_independent_review'}]:
            changed = copy.deepcopy(reviews)
            changed[0].update(patch)
            with self.assertRaises(ValueError):
                report(cases, prior, changed, roster)

    def test_duplicate_unknown_and_wrong_prior_identities_reject(self):
        cases, prior, reviews, roster = self.fixture()
        for changed in [reviews + [reviews[0]], [{**reviews[0], 'qid': 'foreign'}] + reviews[1:]]:
            with self.assertRaises(ValueError):
                report(cases, prior, changed, roster)
        with self.assertRaises(ValueError):
            report(cases, prior[:-1], reviews, roster)
        prior[0]['selection'] = 'disagreement'
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            report(cases, prior, reviews, roster)

    def test_criterion_quotes_cannot_be_copied_from_reference_or_invented(self):
        cases, prior, reviews, roster = self.fixture()
        criterion = {'criterion': 'reference', 'verdict': 'satisfied', 'answer_quote': 'saved answer', 'reason': 'Exact fixture quote.'}
        reviews[0]['criteria'] = [criterion]
        report(cases, prior, reviews, roster)
        for patch in [{'answer_quote': 'reference-only words'}, {'criterion': 'made-up'}, {'verdict': 'uncertain'}, {'verdict': 'violated'}]:
            changed = copy.deepcopy(reviews)
            changed[0]['criteria'][0].update(patch)
            with self.assertRaises(ValueError):
                report(cases, prior, changed, roster)
        reviews[0]['criteria'].append(criterion.copy())
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            report(cases, prior, reviews, roster)

    def test_complete_positive_criteria_cannot_hide_a_negative_overall_label(self):
        cases, prior, reviews, roster = self.fixture()
        reviews[1]['criteria'] = [{'criterion': key, 'verdict': 'satisfied', 'answer_quote': '', 'reason': 'Fixture.'}
                                 for key in ['reference', 'must_include:0', 'must_not_include:0']]
        with self.assertRaisesRegex(ValueError, 'contradicts'):
            report(cases, prior, reviews, roster)

    def test_cli_checks_frozen_packet_hashes_and_never_rewrites_inputs_or_outputs(self):
        cases, prior, reviews, roster = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blobs = {}
            for name, rows in [('blind.jsonl', cases), ('prior-verdicts.jsonl', prior), ('reviews.jsonl', reviews), ('roster.jsonl', roster)]:
                blobs[name] = ''.join(json.dumps(r) + '\n' for r in rows).encode()
                (root / name).write_bytes(blobs[name])
            (root / 'manifest.json').write_text(json.dumps({'protocol': 'memops-judge-audit-selection-v1', 'selected': len(cases), 'outputs_sha256': {name: hashlib.sha256(blobs[name]).hexdigest() for name in ['blind.jsonl', 'prior-verdicts.jsonl']}}))
            command = [sys.executable, str(ROOT / 'python/report_human_calibration.py'), '--packet', str(root), '--reviews', str(root / 'reviews.jsonl'), '--reviewer-roster', str(root / 'roster.jsonl'), '--output', str(root / 'out')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = (root / 'out/summary.json').read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual((root / 'out/summary.json').read_bytes(), summary)
            for name, data in blobs.items():
                self.assertEqual((root / name).read_bytes(), data)
            (root / 'blind.jsonl').write_bytes(blobs['blind.jsonl'] + b' ')
            command[-1] = str(root / 'other')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertIn('hash mismatch', result.stderr)
            self.assertFalse((root / 'other').exists())


if __name__ == '__main__':
    unittest.main()
