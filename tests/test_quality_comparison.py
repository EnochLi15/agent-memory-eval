"""Offline comparison gates must reject biased or incomparable results."""
import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from quality_comparison import EVALUATION_KEYS, compare, interval, load_run, sha, terminal_rows


class QualityComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.data = self.root / 'dataset.json'
        self.put(self.data, [{'sample_id': 's' + str(i), 'benchmark': 'memops',
            'group_id': 'a' if i < 2 else 'b',
            'questions': [{'qid': 'q' + str(i), 'category': 'update'}]} for i in range(4)])
        self.a = self.make('c-sources-memops', [False, False, True, True])
        self.b = self.make('c-rerank-memops', [True, True, False, True], origin=self.a)

    def put(self, path, value):
        path.write_text(json.dumps(value))

    def jsonl(self, path, rows):
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def make(self, name, verdicts, origin=None):
        d = self.root / name
        d.mkdir()
        m = {k: 'fixed' for k in EVALUATION_KEYS}
        m.update(run_id=name, status='finished', planned_questions=4,
            samples=['s' + str(i) for i in range(4)], dataset_sha256=sha(self.data),
            source_state={k: {'dirty': False} for k in ('service', 'eval')},
            service_commit='s1', eval_commit='e1', memory_namespace='same',
            service_configuration={'maxEvidence': 32, 'tokenBudget': 6000, 'rerank': bool(origin)})
        if origin:
            m['service_configuration']['ingestion_origin'] = {'memops': {
                'run_id': origin.name, 'manifest_sha256': sha(origin / 'manifest.json'),
                'service_commit': 's1', 'dataset_sha256': sha(self.data)}}
        self.put(d / 'manifest.json', m)
        rows = [{'qid': 'q' + str(i), 'sample_id': 's' + str(i), 'benchmark': 'memops',
            'category': 'update', 'status': 'judged', 'correct': v} for i, v in enumerate(verdicts)]
        self.jsonl(d / 'judgments.jsonl', rows)
        self.jsonl(d / 'requests.jsonl', [{'path': '/add', 'body': {'request_id': 'chunk'}}])
        self.jsonl(d / 'ingest.jsonl', [{'request_id': 'chunk', 'status': 'ok'}])
        self.audit(d)
        return d

    def audit(self, d):
        self.put(d / 'http-trace-audit.json', {'passed': True, 'run_id': d.name,
            'hashes': {'manifest': sha(d / 'manifest.json'), 'requests': sha(d / 'requests.jsonl'),
                       'dataset': sha(self.data)},
            'counts': {'expected_add_chunks': 1, 'distinct_observed_adds': 1}})

    def load(self, d):
        return load_run(d, {sha(self.data): self.data})

    def cmp(self, kind='query_component'):
        return compare(self.load(self.a), self.load(self.b), kind, self.root)

    def test_failures_remain_in_full_denominator(self):
        p = self.b / 'judgments.jsonl'
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        rows[3] = {**rows[3], 'status': 'pipeline_error'}
        self.jsonl(p, rows)
        r = self.cmp()
        self.assertEqual(r['status'], 'compared')
        self.assertEqual(r['planned'], 4)
        self.assertEqual(r['delta_over_planned'], 0)
        self.assertEqual(r['jointly_judged'], 3)
        self.assertAlmostEqual(r['joint_subset_delta'], 1 / 3)
        self.assertEqual(r['group_sizes'], {'a': 2, 'b': 2})
        self.assertEqual(r['group_delta_bootstrap_95'], [-1, 1])

    def test_missing_question_and_zero_denominator_rejected(self):
        p = self.b / 'judgments.jsonl'
        p.write_text('\n'.join(p.read_text().splitlines()[:-1]))
        with self.assertRaisesRegex(ValueError, 'exactly cover'):
            self.load(self.b)
        p = self.b / 'manifest.json'
        m = json.loads(p.read_text()); m['planned_questions'] = 0; self.put(p, m)
        with self.assertRaisesRegex(ValueError, 'positive integer'):
            self.load(self.b)

    def test_verdict_types_and_revotes_rejected(self):
        for v in ('false', 0, 1, None):
            with self.assertRaises(ValueError):
                terminal_rows([{'qid': 'q', 'status': 'judged', 'correct': v}])
        with self.assertRaises(ValueError):
            terminal_rows([{'qid': 'q', 'status': 'judged', 'correct': False}] * 2)
        self.assertTrue(terminal_rows([{'qid': 'q', 'status': 'judge_error'},
            {'qid': 'q', 'status': 'judged', 'correct': True}])['q']['correct'])

    def test_fixed_answer_budget_and_query_repeat(self):
        left, right = self.load(self.a), self.load(self.b)
        for key in ('answer_model', 'rubric_prompt_sha256', 'top_k'):
            changed = copy.deepcopy(right); changed['manifest'][key] = 'changed'
            self.assertEqual(compare(left, changed, 'query_component', self.root)['status'], 'refused')
        right['manifest']['service_configuration']['tokenBudget'] = 7000
        self.assertEqual(compare(left, right, 'query_component', self.root)['status'], 'refused')
        self.assertIn('Repeat changes', self.cmp('query_repeat')['reason'])
        self.assertIn('distinct runs', compare(left, left, 'query_repeat', self.root)['reason'])

    def test_multiple_query_changes_cannot_be_called_one_component(self):
        left, right = self.load(self.a), self.load(self.b)
        right['manifest']['service_configuration']['eventView'] = True
        result = compare(left, right, 'query_component', self.root)
        self.assertIn('exactly one declared retrieval toggle', result['reason'])
        self.assertEqual(compare(left, right, 'query_bundle', self.root)['status'], 'compared')

    def test_incomplete_adds_rejected_even_with_passed_payload_audit(self):
        p = self.a / 'http-trace-audit.json'
        audit = json.loads(p.read_text()); audit['counts']['expected_add_chunks'] = 2
        self.put(p, audit)
        self.assertIn('every successful planned add', self.cmp()['reason'])

    def test_failed_ingestion_rejected(self):
        self.jsonl(self.a / 'ingest.jsonl', [{'request_id': 'chunk', 'status': 'failed'}])
        self.assertEqual(self.cmp()['status'], 'refused')

    def test_stale_audit_and_origin_reference_rejected(self):
        p = self.a / 'requests.jsonl'; p.write_text(p.read_text() + '\n')
        self.assertIn('stale', self.cmp()['reason'])
        self.audit(self.a)
        p = self.a / 'manifest.json'; p.write_text(p.read_text() + ' ')
        self.assertEqual(self.cmp()['status'], 'refused')

    def test_cluster_bootstrap_and_pending(self):
        self.assertEqual(interval({'a': [0] * 10, 'b': [1] * 10}), [0, 1])
        self.assertIsNone(interval({'a': [0, 1]}))
        self.assertEqual(compare({'status': 'running'}, {'status': 'not_started'},
            'query_component', self.root)['status'], 'pending')
        self.assertEqual(compare({'status': 'invalid'}, {'status': 'ready'},
            'query_component', self.root)['status'], 'refused')

    def test_cli_preserves_absent_matrix_and_rejects_duplicate_pairs(self):
        plan = {'protocol': 'paired-query-quality-plan-v1',
            'profiles': ['sources', 'rerank', 'events'], 'benchmarks': ['memops'],
            'pairs': [{'baseline': 'sources', 'candidate': 'rerank', 'kind': 'query_component'},
                      {'baseline': 'rerank', 'candidate': 'events', 'kind': 'query_component'}]}
        p = self.root / 'plan.json'; self.put(p, plan)
        command = [sys.executable, str(ROOT / 'scripts/compare-quality.py'), '--campaign', 'c',
            '--plan', str(p), '--artifacts', str(self.root), '--data', str(self.data),
            '--output', str(self.root / 'report')]
        r = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads((self.root / 'report/comparison.json').read_text())
        self.assertFalse(report['complete'])
        self.assertEqual(report['planned_runs'], 3)
        self.assertEqual([x['status'] for x in report['comparisons']], ['compared', 'pending'])
        plan['pairs'].append(plan['pairs'][0]); self.put(p, plan)
        r = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('Duplicate or self comparison', r.stderr)


if __name__ == '__main__':
    unittest.main()
