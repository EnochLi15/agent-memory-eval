"""Posthoc source-ID coverage. Gold stays here, outside service/ordinary Answer.

Usage: python3 python/diagnose_retrieval_stages.py --stages PATH --data PATH
       --namespace RUN_NAMESPACE --output PATH
This attributes ID loss, not semantic correctness or whether a filter was proper.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def diagnose(samples, traces, namespace):
    keyed = defaultdict(list)
    for row in traces:
        if row.get('event') == 'retrieval_stages':
            keyed[row['tenant_sha256'], row['query_sha256']].append(row)
    results = []
    for sample in samples:
        user = f"{namespace}:{sample['benchmark']}:{sample['sample_id']}"
        for question in sample['questions']:
            base = {'qid': question['qid'], 'sample_id': sample['sample_id']}
            gold = question.get('gold_evidence') or []
            matches = keyed[digest(user), digest(question['question'])]
            if not gold:
                results.append({**base, 'status': 'no_source_annotation'})
                continue
            if len(matches) != 1:
                results.append({**base, 'status': 'missing_trace' if not matches else 'ambiguous_trace', 'trace_count': len(matches)})
                continue
            row = matches[0]
            sources = {s['id']: set(s['external_ids_sha256']) for s in row['sources']}
            filtered_sources = set(row.get('source_filtered_ids', []))
            eligible = set(row['eligible']) | (set(row.get('source_indexed_ids', [])) - filtered_sources)
            recalled = {item for ids in row['routes'].values() for item in ids}
            reranked = row.get('reranked')
            stage_ids = {
                'indexed': set(sources),
                'eligible': eligible,
                'recalled': recalled,
                'candidate_pool': set(row['candidate_ids']),
                'rerank_selected': {s['id'] for s in reranked if s['score'] > 0} if reranked is not None else set(row['candidate_ids']),
                'packed': set(row['selected']),
            }
            coverage = {stage: set().union(*(sources.get(i, set()) for i in ids)) for stage, ids in stage_ids.items()}
            per_source = []
            for source_id in gold:
                found = {stage: digest(source_id) in values for stage, values in coverage.items()}
                first_missing = next((stage for stage, present in found.items() if not present), None)
                per_source.append({'source_id': source_id, 'stages': found, 'first_missing_stage': first_missing})
            results.append({**base, 'status': 'diagnosed', 'revision': row['revision'], 'sources': per_source,
                            'coverage': {stage: sum(s['stages'][stage] for s in per_source) / len(per_source) for stage in stage_ids}})
    return {'scope': 'posthoc source-ID loss attribution, not semantic entailment or proof of an incorrect lifecycle filter', 'rows': results}


def main():
    parser = argparse.ArgumentParser()
    for arg in ['stages', 'data', 'namespace', 'output']:
        parser.add_argument('--' + arg, required=True)
    args = parser.parse_args()
    data_bytes = Path(args.data).read_bytes()
    stage_bytes = Path(args.stages).read_bytes()
    report = diagnose(json.loads(data_bytes), [json.loads(line) for line in stage_bytes.splitlines() if line.strip()], args.namespace)
    report['inputs'] = {'dataset_sha256': hashlib.sha256(data_bytes).hexdigest(), 'trace_sha256': hashlib.sha256(stage_bytes).hexdigest(), 'namespace': args.namespace}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
