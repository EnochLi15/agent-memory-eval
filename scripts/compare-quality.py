"""Execute an explicit offline comparison plan without omitting unfinished runs."""
import argparse
import json
import pathlib
import re
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'python'))
from quality_comparison import load_run, compare, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True)
    p.add_argument('--plan', required=True, type=pathlib.Path)
    p.add_argument('--artifacts', required=True, type=pathlib.Path)
    p.add_argument('--data', required=True, type=pathlib.Path, action='append')
    p.add_argument('--output', required=True, type=pathlib.Path)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    if plan['protocol'] != 'paired-query-quality-plan-v1':
        raise ValueError('Unknown comparison plan')
    names = [a.campaign, *plan['profiles'], *plan['benchmarks']]
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', n) or n in ('.', '..') for n in names):
        raise ValueError('Invalid planned identifier')
    if len(plan['profiles']) != len(set(plan['profiles'])) or len(plan['benchmarks']) != len(set(plan['benchmarks'])):
        raise ValueError('Duplicate planned profile or benchmark')
    if not plan['profiles'] or not plan['benchmarks'] or not plan['pairs']:
        raise ValueError('Empty comparison plan')
    if any(x[k] not in plan['profiles'] for x in plan['pairs'] for k in ('baseline', 'candidate')):
        raise ValueError('Comparison references an unplanned profile')
    pair_ids = [(x['baseline'], x['candidate']) for x in plan['pairs']]
    if len(set(pair_ids)) != len(pair_ids) or any(a == b for a, b in pair_ids):
        raise ValueError('Duplicate or self comparison')
    if any(x['kind'] not in ('query_component', 'query_bundle', 'query_repeat') for x in plan['pairs']):
        raise ValueError('Unknown comparison kind')
    datasets = {sha(path): path for path in a.data}
    runs, pairs = {}, []
    for profile in plan['profiles']:
        for benchmark in plan['benchmarks']:
            run_id = f'{a.campaign}-{profile}-{benchmark}'
            try:
                run = load_run(a.artifacts / run_id, datasets)
                if run['status'] == 'ready' and {q['benchmark'] for q in run['questions'].values()} != {benchmark}:
                    raise ValueError('Dataset benchmark differs from plan')
                runs[profile, benchmark] = run
            except (ValueError, KeyError, FileNotFoundError) as error:
                runs[profile, benchmark] = {'run_id': run_id, 'status': 'invalid', 'reason': str(error)}
    for pair in plan['pairs']:
        for benchmark in plan['benchmarks']:
            result = compare(runs[pair['baseline'], benchmark], runs[pair['candidate'], benchmark], pair['kind'], a.artifacts)
            pairs.append({**pair, 'benchmark': benchmark, **result})
    summary = {'protocol': 'explicit-query-quality-comparison-v1', 'campaign': a.campaign,
               'plan_sha256': sha(a.plan), 'script_sha256': sha(pathlib.Path(__file__)),
               'module_sha256': sha(pathlib.Path(__file__).resolve().parents[1] / 'python/quality_comparison.py'),
               'planned_runs': len(runs), 'planned_comparisons': len(pairs),
               'complete': all(r['status'] == 'ready' for r in runs.values()) and all(r['status'] == 'compared' for r in pairs),
               'runs': [{k: v for k, v in r.items() if k in ('run_id', 'status', 'reason', 'hashes')} for r in runs.values()],
               'comparisons': pairs, 'scope': 'All planned runs and comparisons remain visible, including pending and refused. No service/model calls; no replacement of original artifacts or human calibration.'}
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'complete': summary['complete'], 'planned_runs': len(runs), 'planned_comparisons': len(pairs), 'statuses': {s: sum(r['status'] == s for r in pairs) for s in sorted({r['status'] for r in pairs})}}))


if __name__ == '__main__':
    main()
