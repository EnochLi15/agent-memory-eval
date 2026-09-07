"""Audit recorded execution exposure without reading candidate dataset contents.

Exclusion is by MemOps background group, covering every operation variant.
No recorded exposure is not a guarantee of no manual inspection outside logs.
"""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

IDENTITY = re.compile(r'(?<![A-Za-z0-9])([A-F]\d{2})_(?:remember|forget|reflect|update|trajectory_ops)(?![A-Za-z_])')
LOG_NAMES = {'manifest.json', 'requests.jsonl', 'predictions.jsonl', 'results.jsonl', 'ingest.jsonl', 'blind.jsonl', 'extraction-proposals.jsonl', 'results.json'}


def audit(source, logs, manual=()):
    files = sorted(source.glob('*.json'))
    groups = defaultdict(list)
    for p in files:
        match = IDENTITY.fullmatch(p.stem)
        if not match:
            raise ValueError('Unknown source identity: ' + p.name)
        groups[match[1]].append(p.name)
    evidence = defaultdict(set)
    hashes = {}
    for p in sorted(set(logs)):
        body = p.read_bytes()
        hashes[str(p)] = hashlib.sha256(body).hexdigest()
        for match in IDENTITY.finditer(body.decode('utf-8')):
            evidence[match[1]].add(str(p))
    for group in manual:
        if group not in groups:
            raise ValueError('Unknown manually exposed background group')
        evidence[group].add('explicit_manual_exposure')
    names = '\n'.join(p.name for p in files)
    return {'protocol': 'recorded-background-exposure-v1', 'candidate_contents_read': False,
            'source_file_count': len(files), 'source_group_count': len(groups),
            'inventory_names_sha256': hashlib.sha256(names.encode()).hexdigest(),
            'recorded_exposed_groups': sorted(set(groups) & set(evidence)),
            'no_recorded_exposure_groups': sorted(set(groups) - set(evidence)),
            'excluded_files': [f for g in sorted(set(groups) & set(evidence)) for f in groups[g]],
            'no_recorded_exposure_files': [f for g in sorted(set(groups) - set(evidence)) for f in groups[g]],
            'evidence': {g: sorted(paths) for g, paths in sorted(evidence.items())},
            'input_sha256': hashes,
            'limitations': ['Only the supplied execution logs and explicit manual exclusions are audited.',
                            'Unlogged manual inspection and model pretraining exposure cannot be ruled out.',
                            'No candidate questions are selected, inspected, or evaluated by this audit.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--logs', type=Path, action='append', required=True)
    parser.add_argument('--manual-exposed', nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    logs = [p for root in args.logs for p in root.rglob('*') if p.is_file() and p.name in LOG_NAMES]
    report = audit(args.source, logs, args.manual_exposed)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({key: report[key] for key in ('source_file_count', 'source_group_count')} | {
        'logs_audited': len(logs), 'recorded_exposed_groups': len(report['recorded_exposed_groups']),
        'no_recorded_exposure_groups': len(report['no_recorded_exposure_groups'])}))


if __name__ == '__main__':
    main()
