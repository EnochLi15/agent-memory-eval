"""Strict paired query comparisons; reads artifacts only, never calls a model."""
import collections
import copy
import hashlib
import json
import random
from pathlib import Path

EVALUATION_KEYS = ('dataset_sha256', 'planned_questions', 'samples', 'answer_model',
                   'answer_inference', 'answer_base', 'judge_model', 'judge_kind',
                   'judge_base', 'answer_prompt_sha256', 'rubric_prompt_sha256',
                   'top_k', 'chunk_messages', 'chunk_words')
QUERY_FIELDS = {'host', 'port', 'rawFallback', 'rerank', 'coveragePacking',
                'candidateLimit', 'rerankCandidates', 'retrieval', 'eventView',
                'searchTimeout', 'relationMode', 'rerankPolicy', 'rerankFormat',
                'ingestion_origin'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def terminal_rows(records):
    result = {}
    for row in records:
        if row.get('status') not in ('judged', 'service_error', 'judge_error', 'pipeline_error'):
            raise ValueError('Unknown terminal question status')
        if row['status'] == 'judged' and type(row.get('correct')) is not bool:
            raise ValueError('Judged result requires a boolean verdict')
        if row['qid'] in result and result[row['qid']]['status'] == 'judged':
            raise ValueError('A completed judgment cannot be replaced or voted again')
        result[row['qid']] = row
    return result


def load_run(directory, datasets):
    directory = Path(directory)
    if not (directory / 'manifest.json').exists():
        return {'status': 'not_started', 'run_id': directory.name}
    m = read(directory / 'manifest.json')
    if m['status'] != 'finished':
        return {'status': m['status'], 'run_id': directory.name}
    if m['run_id'] != directory.name:
        raise ValueError('Run directory identity differs from manifest')
    data_path = datasets.get(m['dataset_sha256'])
    if data_path is None or sha(data_path) != m['dataset_sha256']:
        raise ValueError('Exact dataset is required')
    data = read(data_path)
    samples = {s['sample_id']: s for s in data}
    if len(samples) != len(data) or len(m['samples']) != len(set(m['samples'])):
        raise ValueError('Repeated sample identity')
    if type(m.get('planned_questions')) is not int or m['planned_questions'] <= 0:
        raise ValueError('Planned question count must be a positive integer')
    expected, remaining = {}, m['planned_questions']
    for sid in m['samples']:
        sample = samples[sid]
        if not isinstance(sample.get('group_id'), str) or not sample['group_id']:
            raise ValueError('Dataset has no declared background group')
        for q in sample['questions'][:remaining]:
            if q['qid'] in expected:
                raise ValueError('Repeated dataset question')
            expected[q['qid']] = {'sample_id': sid, 'benchmark': sample['benchmark'],
                                  'group_id': sample['group_id'], 'category': q['category']}
        remaining -= min(remaining, len(sample['questions']))
    if len({q['benchmark'] for q in expected.values()}) != 1:
        raise ValueError('A run must select exactly one benchmark')
    rows = terminal_rows(lines(directory / 'judgments.jsonl'))
    if remaining or set(rows) != set(expected):
        raise ValueError('Terminal questions do not exactly cover planned dataset selection')
    if any(any(row.get(k) != expected[qid][k] for k in ('sample_id', 'benchmark', 'category'))
           for qid, row in rows.items()):
        raise ValueError('Question identity/category differs from dataset')
    return {'status': 'ready', 'run_id': m['run_id'], 'manifest': m, 'rows': rows,
            'questions': expected, 'directory': directory,
            'hashes': {name: sha(directory / name) for name in ('manifest.json', 'judgments.jsonl')},
            'dataset_path': str(data_path)}


def score(row):
    return int(row['status'] == 'judged' and row['correct'])


def interval(groups):
    if len(groups) < 2:
        return None
    units = [groups[k] for k in sorted(groups)]
    rng = random.Random(20260905)
    draws = []
    for _ in range(1000):
        values = [value for _ in units for value in rng.choice(units)]
        draws.append(sum(values) / len(values))
    draws.sort()
    return [draws[25], draws[975]]


def write_config(config):
    value = copy.deepcopy(config)
    for field in QUERY_FIELDS:
        value.pop(field, None)
    if isinstance(value.get('experimental'), dict):
        value['experimental'].pop('multiHop', None)
    return value


def origin_proof(run, benchmark, artifacts):
    m = run['manifest']
    reference = m['service_configuration'].get('ingestion_origin', {}).get(benchmark)
    origin_id = reference['run_id'] if reference else m['run_id']
    # Origin IDs are artifact directory names, never arbitrary filesystem paths.
    if not isinstance(origin_id, str) or not origin_id or Path(origin_id).name != origin_id or '\\' in origin_id or origin_id in ('.', '..'):
        raise ValueError('Invalid ingestion origin identity')
    directory = Path(artifacts) / origin_id
    origin = read(directory / 'manifest.json')
    if reference and reference['manifest_sha256'] != sha(directory / 'manifest.json'):
        raise ValueError('Ingestion origin hash changed')
    if origin.get('run_id') != origin_id:
        raise ValueError('Origin directory identity differs from manifest')
    if origin['status'] != 'finished':
        raise ValueError('Ingestion origin is unfinished')
    for item in (m, origin):
        if any(item.get('source_state', {}).get(k, {}).get('dirty') is not False
               or not item.get(k + '_commit') for k in ('service', 'eval')):
            raise ValueError('Shared ingestion needs clean recorded source identities')
    if reference and any(reference.get(k) != origin.get(k) for k in ('service_commit', 'dataset_sha256')):
        raise ValueError('Origin reference identity differs from manifest')
    for key in ('service_commit', 'eval_commit', 'dataset_sha256', 'memory_namespace'):
        if origin.get(key) != m.get(key):
            raise ValueError('Origin identity mismatch: ' + key)
    if write_config(origin['service_configuration']) != write_config(m['service_configuration']):
        raise ValueError('Write configuration differs from ingestion origin')
    audit = read(directory / 'http-trace-audit.json')
    if audit.get('passed') is not True or audit.get('run_id') != origin_id:
        raise ValueError('Complete ingestion requires a passed service-payload audit')
    for key, file in (('manifest', 'manifest.json'), ('requests', 'requests.jsonl')):
        if audit['hashes'][key] != sha(directory / file):
            raise ValueError('Service-payload audit is stale')
    if audit['hashes']['dataset'] != origin['dataset_sha256']:
        raise ValueError('Service-payload audit used a different dataset')
    counts = audit['counts']
    last = {r['request_id']: r for r in lines(directory / 'ingest.jsonl')}
    observed = {r['body']['request_id'] for r in lines(directory / 'requests.jsonl') if r['path'] == '/add'}
    if not observed or counts['expected_add_chunks'] != len(observed) or counts['distinct_observed_adds'] != len(observed) or set(last) != observed or any(r['status'] != 'ok' for r in last.values()):
        raise ValueError('Ingestion does not contain every successful planned add')
    return {'run_id': origin_id, 'manifest_sha256': sha(directory / 'manifest.json'),
            'ingest_sha256': sha(directory / 'ingest.jsonl'), 'audit_sha256': sha(directory / 'http-trace-audit.json')}


def compare(left, right, kind, artifacts):
    if left['status'] == 'invalid' or right['status'] == 'invalid':
        return {'status': 'refused', 'reason': 'Invalid run artifacts', 'left_reason': left.get('reason'), 'right_reason': right.get('reason')}
    if left['status'] != 'ready' or right['status'] != 'ready':
        return {'status': 'pending', 'left_status': left['status'], 'right_status': right['status']}
    a, b = left['manifest'], right['manifest']
    mismatch = [k for k in EVALUATION_KEYS if k not in a or k not in b or a[k] != b[k]]
    if mismatch or left['questions'] != right['questions']:
        return {'status': 'refused', 'reason': 'Evaluation or exact question identity differs', 'fields': mismatch}
    try:
        if kind not in ('query_component', 'query_bundle', 'query_repeat'):
            raise ValueError('Unknown comparison kind')
        if left['run_id'] == right['run_id']:
            raise ValueError('Comparison requires distinct runs')
        ca, cb = a['service_configuration'], b['service_configuration']
        if any(k not in ca or k not in cb or ca[k] != cb[k] for k in ('maxEvidence', 'tokenBudget')):
            raise ValueError('Final evidence budgets differ')
        if write_config(ca) != write_config(cb):
            raise ValueError('Write configuration differs')
        clean = lambda c: {k: v for k, v in c.items() if k not in ('host', 'port', 'ingestion_origin')}
        qa, qb = clean(ca), clean(cb)
        changed = sorted(k for k in set(qa) | set(qb) if qa.get(k) != qb.get(k))
        if kind == 'query_repeat' and changed:
            raise ValueError('Repeat changes query configuration')
        if kind == 'query_component' and (len(changed) != 1 or changed[0] not in ('rerank', 'coveragePacking', 'eventView')):
            raise ValueError('Component comparison must change exactly one declared retrieval toggle')
        benchmark = next(iter(left['questions'].values()))['benchmark']
        origins = [origin_proof(r, benchmark, artifacts) for r in (left, right)]
        if origins[0] != origins[1]:
            raise ValueError('Runs do not share the same complete ingestion')
    except (ValueError, KeyError, FileNotFoundError) as error:
        return {'status': 'refused', 'reason': str(error)}
    groups, changes, joint = collections.defaultdict(list), [], []
    for qid, identity in left['questions'].items():
        before, after = left['rows'][qid], right['rows'][qid]
        old, new = score(before), score(after)
        groups[identity['group_id']].append(new - old)
        if before['status'] == after['status'] == 'judged':
            joint.append(new - old)
        if old != new or before['status'] != after['status']:
            changes.append({'qid': qid, **identity, 'before_status': before['status'],
                            'after_status': after['status'], 'before_score': old, 'after_score': new})
    values = [x for unit in groups.values() for x in unit]
    return {'status': 'compared', 'kind': kind, 'planned': len(values), 'delta_over_planned': sum(values) / len(values),
            'changed_configuration_fields': changed,
            'correct_over_planned': [sum(score(r) for r in x['rows'].values()) / len(values) for x in (left, right)],
            'group_units': len(groups), 'group_sizes': {k: len(v) for k, v in groups.items()},
            'group_delta_bootstrap_95': interval(groups), 'changes': changes,
            'wrong_to_correct': values.count(1), 'correct_to_wrong': values.count(-1),
            'jointly_judged': len(joint), 'joint_subset_delta': sum(joint) / len(joint) if joint else None,
            'status_counts': [dict(collections.Counter(r['status'] for r in x['rows'].values())) for x in (left, right)],
            'origin': origins[0], 'hashes': [left['hashes'], right['hashes']],
            'scope': 'Errors remain non-correct in the full planned denominator. Joint subset is conditional and potentially biased. Background variants are resampled together; query_bundle cannot isolate one component. No human truth or generalization claim.'}
