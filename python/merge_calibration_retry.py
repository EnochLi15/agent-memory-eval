"""Merge transport-only retries without resampling semantic verdicts or dropping cases."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keyed(items):
    result = {}
    for row in items:
        if row['qid'] in result:
            raise ValueError('Duplicate qid')
        result[row['qid']] = row
    return result


def merge(original, retry):
    old, new = keyed(original), keyed(retry)
    expected = {qid for qid, r in old.items() if r['status'] == 'error'}
    if set(new) != expected:
        raise ValueError('Retry must contain exactly the original transport failures')
    return [{**(new[qid] if qid in new else row),
             'audit_origin': 'transport_retry' if qid in new else 'original'}
            for qid, row in old.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'retry', 'packet', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    old_manifest = json.loads((args.original / 'manifest.json').read_text())
    new_manifest = json.loads((args.retry / 'manifest.json').read_text())
    old_path, new_path = args.original / 'results.jsonl', args.retry / 'results.jsonl'
    for manifest, path in ((old_manifest, old_path), (new_manifest, new_path)):
        if manifest['results_sha256'] != sha(path):
            raise ValueError('Results checksum mismatch')
    if new_manifest['input_provenance']['parent_results_sha256'] != sha(old_path):
        raise ValueError('Retry belongs to another parent')
    for key in ('protocol', 'prompt_sha256', 'source_sha256', 'model_runtime_sha256', 'model', 'base', 'inference'):
        if old_manifest[key] != new_manifest[key]:
            raise ValueError('Retry protocol changed: ' + key)
    cases = keyed(rows(args.packet / 'blind.jsonl'))
    comparisons = keyed(rows(args.packet / 'prior-verdicts.jsonl'))
    merged = merge(rows(old_path), rows(new_path))
    if len(merged) != old_manifest['planned'] or set(keyed(merged)) != set(cases) or set(cases) != set(comparisons):
        raise ValueError('Packet or planned denominator mismatch')
    by_type, compared = defaultdict(Counter), defaultdict(Counter)
    pending = []
    for row in merged:
        by_type[row['evaluation_type']][row['status']] += 1
        if row['status'] == 'judged':
            for judge in ('primary_correct', 'upstream_correct'):
                compared[judge][f"prior_{comparisons[row['qid']][judge]}_audit_{row['correct']}"] += 1
        else:
            pending.append({**cases[row['qid']], 'audit': row, 'review_status': 'pending'})
    summary = {'protocol': 'transport-only-audit-merge-v1', 'audit_protocol': old_manifest['protocol'],
               'planned': len(merged), 'status_counts': dict(Counter(r['status'] for r in merged)),
               'correct': sum(r.get('correct') is True for r in merged),
               'incorrect': sum(r.get('correct') is False for r in merged),
               'by_type': dict(by_type), 'prior_vs_audit': dict(compared),
               'independent_human_reviews': 0,
               'scope': 'Same-model posthoc criterion audit on selected exposed answers. No historical score replacement; unresolved rows remain in denominator.'}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, values in (('results.jsonl', merged), ('pending-review.jsonl', pending)):
        (args.output / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest = {'protocol': summary['protocol'], 'original': str(args.original), 'retry': str(args.retry),
                'input_hashes': {str(p): sha(p) for p in (old_path, new_path, args.original/'manifest.json', args.retry/'manifest.json', args.packet/'blind.jsonl', args.packet/'prior-verdicts.jsonl')},
                'source_sha256': sha(Path(__file__)), 'results_sha256': sha(args.output/'results.jsonl')}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
