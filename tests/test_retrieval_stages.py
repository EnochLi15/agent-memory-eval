import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('stages', Path(__file__).parents[1] / 'python/diagnose_retrieval_stages.py')
stages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stages)


class StageDiagnosis(unittest.TestCase):
    def fixture(self):
        samples = [{'sample_id': 's', 'benchmark': 'locomo', 'questions': [{'qid': 'q', 'question': 'Where?', 'gold_evidence': ['D1:1', 'D1:2']}]}]
        trace = {'event': 'retrieval_stages', 'tenant_sha256': stages.digest('run:locomo:s'), 'query_sha256': stages.digest('Where?'), 'revision': 5,
                 'sources': [{'id': 'a', 'external_ids_sha256': [stages.digest('D1:1')]}, {'id': 'b', 'external_ids_sha256': [stages.digest('D1:2')]}],
                 'eligible': ['a', 'b'], 'routes': {'lexical': ['a', 'b']}, 'candidate_ids': ['a', 'b'], 'reranked': [{'id': 'a', 'score': 1}, {'id': 'b', 'score': 0}], 'selected': ['a']}
        return samples, trace

    def test_rerank_loss_is_distinguished_from_packing(self):
        samples, trace = self.fixture()
        row = stages.diagnose(samples, [trace], 'run')['rows'][0]
        self.assertEqual(row['sources'][1]['first_missing_stage'], 'rerank_selected')
        self.assertIsNone(row['sources'][0]['first_missing_stage'])
        trace['reranked'][1]['score'] = .8
        row = stages.diagnose(samples, [trace], 'run')['rows'][0]
        self.assertEqual(row['sources'][1]['first_missing_stage'], 'packed')

    def test_ambiguous_missing_and_unannotated_are_not_scored_as_misses(self):
        samples, trace = self.fixture()
        self.assertEqual(stages.diagnose(samples, [trace, trace], 'run')['rows'][0]['status'], 'ambiguous_trace')
        self.assertEqual(stages.diagnose(samples, [trace], 'other')['rows'][0]['status'], 'missing_trace')
        samples[0]['questions'][0]['gold_evidence'] = []
        self.assertEqual(stages.diagnose(samples, [trace], 'run')['rows'][0]['status'], 'no_source_annotation')


if __name__ == '__main__':
    unittest.main()
