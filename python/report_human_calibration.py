"""Validate submitted human labels and compare the two frozen prior judges.

No models are called, no missing labels are inferred and no historical scores
are changed. Human identity and independence are self-attested, not authenticated
by this offline tool. The selected disagreement-heavy sample is not a population
estimate of judge accuracy.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def keyed(rows, field):
    result = {}
    for row in rows:
        identity = row.get(field)
        if not isinstance(identity, str) or not identity.strip() or identity in result:
            raise ValueError('Missing or duplicate ' + field)
        result[identity] = row
    return result


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate_review(case, row):
    status = row.get('status')
    if status == 'pending_independent_review':
        if row.get('correct') is not None or row.get('criteria') or row.get('reason'):
            raise ValueError('Pending row contains a verdict or rationale')
        return
    if status not in ('reviewed', 'uncertain'):
        raise ValueError('Unknown human review status')
    if row.get('reviewer_kind') != 'human' or not nonempty(row.get('reviewer')):
        raise ValueError('Completed review requires a declared human reviewer')
    if not nonempty(row.get('reason')):
        raise ValueError('Completed review requires a reason')
    if status == 'reviewed' and type(row.get('correct')) is not bool:
        raise ValueError('Reviewed label must be a boolean')
    if status == 'uncertain' and row.get('correct') is not None:
        raise ValueError('Uncertain review cannot contain a binary verdict')
    criteria = row.get('criteria', [])
    if not isinstance(criteria, list):
        raise ValueError('Criteria must be an array')
    allowed = {'reference'}
    for key in ('must_include', 'must_not_include', 'harmful_extra'):
        rules = case['rubric'].get(key, [])
        if not isinstance(rules, list):
            raise ValueError('Malformed rubric')
        allowed.update(f'{key}:{i}' for i in range(len(rules)))
    seen, verdicts = set(), []
    for item in criteria:
        if not isinstance(item, dict):
            raise ValueError('Malformed human criterion')
        criterion = item.get('criterion')
        if criterion not in allowed or criterion in seen:
            raise ValueError('Unknown or duplicate human criterion')
        seen.add(criterion)
        verdict = item.get('verdict')
        if verdict not in ('satisfied', 'violated', 'uncertain') or not nonempty(item.get('reason')):
            raise ValueError('Invalid human criterion verdict or reason')
        quote = item.get('answer_quote', '')
        if not isinstance(quote, str) or quote and quote not in case['answer']:
            raise ValueError('Human quote is not an exact saved-answer substring')
        verdicts.append(verdict)
    if status == 'reviewed':
        if 'uncertain' in verdicts or row['correct'] and 'violated' in verdicts:
            raise ValueError('Overall human verdict contradicts supplied criteria')
        if seen == allowed and not row['correct'] and all(v == 'satisfied' for v in verdicts):
            raise ValueError('Negative human label contradicts complete satisfied criteria')


def confusion(rows, judge):
    matrix = Counter({'true_positive': 0, 'true_negative': 0, 'false_positive': 0, 'false_negative': 0})
    for row in rows:
        human, predicted = row['human_correct'], row[judge + '_correct']
        key = ('true_positive' if human else 'false_positive') if predicted else ('false_negative' if human else 'true_negative')
        matrix[key] += 1
    n = len(rows)
    return {**matrix, 'compared': n, 'agreement': (matrix['true_positive'] + matrix['true_negative']) / n if n else None}


def report(blind, prior, reviews, roster=()):
    cases, previous, labels, people = keyed(blind, 'qid'), keyed(prior, 'qid'), keyed(reviews, 'qid'), keyed(roster, 'reviewer')
    if set(previous) != set(cases) or set(labels) - set(cases):
        raise ValueError('Review/prior identities do not match frozen packet')
    for person in people.values():
        if person.get('reviewer_kind') != 'human' or person.get('independent_of_model_judging') is not True or not nonempty(person.get('attestation')):
            raise ValueError('Reviewer roster requires explicit human independence attestation')
    accepted, status, unconfirmed = [], Counter(), set()
    for qid, case in cases.items():
        old = previous[qid]
        if type(old.get('primary_correct')) is not bool or type(old.get('upstream_correct')) is not bool or old.get('selection') not in ('agreement', 'disagreement'):
            raise ValueError('Invalid frozen prior verdict')
        if (old['primary_correct'] == old['upstream_correct']) != (old['selection'] == 'agreement'):
            raise ValueError('Prior selection conflicts with frozen verdicts')
        row = labels.get(qid)
        if row is None:
            status['missing_review'] += 1
            continue
        validate_review(case, row)
        status[row['status']] += 1
        if row['status'] in ('reviewed', 'uncertain') and row['reviewer'] not in people:
            unconfirmed.add(row['reviewer'])
        if row['status'] == 'reviewed':
            accepted.append({'qid': qid, 'evaluation_type': case['evaluation_type'], 'selection': old['selection'],
                             'human_correct': row['correct'], 'primary_correct': old['primary_correct'],
                             'upstream_correct': old['upstream_correct'],
                             'independence_attested': row['reviewer'] in people})
    confirmed = [r for r in accepted if r['independence_attested']]
    compare = lambda rows: {judge: confusion(rows, judge) for judge in ('primary', 'upstream')}
    complete = len(accepted) == len(cases) and bool(cases) and not unconfirmed
    summary = {
        'protocol': 'human-calibration-comparison-v1', 'selected_cases': len(cases),
        'review_status_counts': dict(status), 'validated_binary_human_labels': len(accepted),
        'reviewers_without_independence_attestation': len(unconfirmed),
        'compared_independent_labels': len(confirmed),
        'status': 'ready_for_calibration_review' if complete else 'incomplete_independent_review',
        'overall': compare(confirmed),
        'by_question_type': {kind: compare([r for r in confirmed if r['evaluation_type'] == kind]) for kind in sorted({c['evaluation_type'] for c in cases.values()})},
        'by_selection': {kind: compare([r for r in confirmed if r['selection'] == kind]) for kind in ('agreement', 'disagreement')},
        'coverage_by_question_type': {kind: {'selected': sum(c['evaluation_type'] == kind for c in cases.values()), 'reviewed': sum(r['evaluation_type'] == kind for r in accepted), 'compared': sum(r['evaluation_type'] == kind for r in confirmed)} for kind in sorted({c['evaluation_type'] for c in cases.values()})},
        'scope': ['Submitted reviewer identity/independence is self-attested, not authenticated by this tool.',
                  'Only explicitly reviewed binary labels with a roster independence attestation enter confusion matrices; uncertainty, unconfirmed reviewers and missing labels remain open.',
                  'Disagreements are oversampled. Selected-packet agreement is not a population judge-accuracy estimate.',
                  'Optional criterion quotes receive literal provenance checks, not automatic semantic adjudication.',
                  'No historical benchmark score, blind packet, prior verdict or submitted label is rewritten.'],
    }
    return summary, accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--reviewer-roster', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.packet / 'manifest.json').read_text())
    if manifest.get('protocol') != 'memops-judge-audit-selection-v1':
        raise ValueError('Unsupported frozen selection protocol')
    inputs = {'blind': args.packet / 'blind.jsonl', 'prior': args.packet / 'prior-verdicts.jsonl', 'reviews': args.reviews}
    if args.reviewer_roster:
        inputs['roster'] = args.reviewer_roster
    raw = {name: path.read_bytes() for name, path in inputs.items()}
    for name, filename in (('blind', 'blind.jsonl'), ('prior', 'prior-verdicts.jsonl')):
        if digest(raw[name]) != manifest['outputs_sha256'][filename]:
            raise ValueError('Frozen packet hash mismatch: ' + filename)
    rows = {name: [json.loads(line) for line in data.splitlines() if line.strip()] for name, data in raw.items()}
    if len(rows['blind']) != manifest.get('selected'):
        raise ValueError('Selected case count differs from frozen manifest')
    summary, accepted = report(rows['blind'], rows['prior'], rows['reviews'], rows.get('roster', []))
    summary['input_sha256'] = {name: digest(data) for name, data in raw.items()}
    summary['packet_manifest_sha256'] = digest((args.packet / 'manifest.json').read_bytes())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    (args.output / 'validated-comparisons.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in accepted))
    print(json.dumps({k: summary[k] for k in ('status', 'selected_cases', 'review_status_counts', 'validated_binary_human_labels')}))


if __name__ == '__main__':
    main()
