"""Supplement saved scores with dataset-group bootstrap intervals; no model calls."""
import argparse
import collections
import hashlib
import json
import pathlib
import random
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--run-id', required=True)
parser.add_argument('--data', required=True)
parser.add_argument('--compare-run')
args = parser.parse_args()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(run_id):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', run_id):
        raise ValueError('Invalid run ID')
    directory = ROOT / 'artifacts' / run_id
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['status'] != 'finished':
        raise ValueError('Only finished runs may be analyzed')
    rows = {r['qid']: r for r in map(json.loads, (directory / 'judgments.jsonl').read_text().splitlines())}
    return directory, manifest, rows


def interval(groups):
    rng = random.Random(20260905)
    units = list(groups.values())
    if not units:
        raise ValueError('Empty grouping')
    draws = []
    for _ in range(1000):
        sampled = [rng.choice(units) for _ in units]
        values = [value for unit in sampled for value in unit]
        draws.append(sum(values) / len(values))
    draws.sort()
    return [draws[25], draws[975]]


directory, manifest, judgments = read_run(args.run_id)
data_path = pathlib.Path(args.data)
if sha(data_path) != manifest['dataset_sha256']:
    raise ValueError('Dataset hash differs from evaluated data')
samples = {s['sample_id']: s for s in json.loads(data_path.read_text())}
qid_groups = {}
remaining = manifest['planned_questions']
for sample_id in manifest['samples']:
    sample = samples[sample_id]
    for question in sample['questions'][:remaining]:
        if question['qid'] in qid_groups:
            raise ValueError('Repeated question ID')
        qid_groups[question['qid']] = sample['group_id']
    remaining -= min(remaining, len(sample['questions']))
if remaining or set(qid_groups) != set(judgments):
    raise ValueError('Finished judgment IDs do not exactly match the planned dataset selection')
score = lambda row: int(row['status'] == 'judged' and bool(row.get('correct')))
groups = collections.defaultdict(list)
for qid, group in qid_groups.items():
    groups[group].append(score(judgments[qid]))
result = {'run_id': args.run_id, 'planned': len(qid_groups),
          'judged': sum(r['status'] == 'judged' for r in judgments.values()),
          'correct': sum(score(r) for r in judgments.values()),
          'group_units': len(groups), 'group_question_counts': {g: len(v) for g, v in groups.items()},
          'group_bootstrap_95': interval(groups), 'seed': 20260905, 'draws': 1000,
          'resampling': 'Python random.Random; sample dataset group_id units with replacement, retaining all questions in each selected unit; question-weighted accuracy.',
          'scope': 'Supplement to original per-sample metrics. Dataset group_id binds MemOps background variants together. Errors remain non-correct in the planned denominator; not an official competition score.',
          'hashes': {'dataset': sha(data_path), 'manifest': sha(directory / 'manifest.json'),
                     'judgments': sha(directory / 'judgments.jsonl'), 'script': sha(pathlib.Path(__file__))}}
if args.compare_run:
    other_dir, other_manifest, other_rows = read_run(args.compare_run)
    keys = ['dataset_sha256', 'planned_questions', 'samples', 'answer_model', 'answer_inference',
            'answer_base', 'judge_model', 'judge_kind', 'judge_base', 'answer_prompt_sha256',
            'rubric_prompt_sha256', 'top_k', 'chunk_messages', 'chunk_words']
    if any(manifest.get(k) != other_manifest.get(k) for k in keys) or set(other_rows) != set(judgments):
        raise ValueError('Comparison does not share data and evaluation configuration')
    changes = collections.defaultdict(list)
    for qid, group in qid_groups.items():
        changes[group].append(score(judgments[qid]) - score(other_rows[qid]))
    values = [v for group in changes.values() for v in group]
    result['paired'] = {'baseline': args.compare_run, 'candidate': args.run_id,
                        'delta': sum(values) / len(values), 'group_delta_bootstrap_95': interval(changes),
                        'wrong_to_correct': values.count(1), 'correct_to_wrong': values.count(-1),
                        'baseline_manifest_sha256': sha(other_dir / 'manifest.json'),
                        'baseline_judgments_sha256': sha(other_dir / 'judgments.jsonl')}
output = directory / 'grouped-statistics'
output.mkdir(exist_ok=True)
name = 'summary.json' if not args.compare_run else 'vs-' + args.compare_run + '.json'
(output / name).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
