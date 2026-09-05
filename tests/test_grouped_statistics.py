"""Regression for correlated samples and the statistics provenance guard."""
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]


class GroupedStatisticsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = pathlib.Path(self.temp.name) / 'data.json'
        samples = [{'sample_id': f's{i}', 'group_id': 'a' if i < 10 else 'b',
                    'questions': [{'qid': f'q{i}'}]} for i in range(20)]
        self.data.write_text(json.dumps(samples))
        self.run_id = 'group-stats-fixture-' + uuid.uuid4().hex
        self.directory = ROOT / 'artifacts' / self.run_id
        self.directory.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.directory)
        manifest = {'run_id': self.run_id, 'status': 'finished', 'planned_questions': 20,
                    'samples': [s['sample_id'] for s in samples],
                    'dataset_sha256': hashlib.sha256(self.data.read_bytes()).hexdigest()}
        (self.directory / 'manifest.json').write_text(json.dumps(manifest))
        rows = [{'qid': f'q{i}', 'sample_id': f's{i}', 'status': 'judged', 'correct': i < 10}
                for i in range(20)]
        (self.directory / 'judgments.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def call(self):
        return subprocess.run([sys.executable, str(ROOT / 'scripts/grouped-statistics.py'),
                               '--run-id', self.run_id, '--data', str(self.data)],
                              capture_output=True, text=True, timeout=20)

    def test_correlated_backgrounds_are_resampled_as_whole_units(self):
        result = self.call()
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['correct'], 10)
        self.assertEqual(data['group_units'], 2)
        self.assertEqual(data['group_question_counts'], {'a': 10, 'b': 10})
        # Twenty independent Bernoulli samples would give a much narrower interval.
        self.assertEqual(data['group_bootstrap_95'], [0, 1])

    def test_modified_dataset_cannot_rewrite_statistics(self):
        self.assertEqual(self.call().returncode, 0)
        output = self.directory / 'grouped-statistics/summary.json'
        previous = output.read_bytes()
        self.data.write_text(self.data.read_text() + ' ')
        result = self.call()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Dataset hash differs', result.stderr)
        self.assertEqual(output.read_bytes(), previous)
