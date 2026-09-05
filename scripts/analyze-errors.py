"""Evidence-bound error inventory from saved HTTP evaluation artifacts."""
import argparse,collections,hashlib,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];directory=root/'artifacts'/a.run_id
if directory.parent!=root/'artifacts':raise SystemExit('Invalid run ID')
manifest=json.loads((directory/'manifest.json').read_text())
if manifest['status']!='finished':raise SystemExit('Only finished runs may be analyzed')
def rows(name):
 path=directory/name
 return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
judgments={r['qid']:r for r in rows('judgments.jsonl')};retrievals={r['qid']:r for r in rows('retrievals.jsonl')};predictions={r['qid']:r for r in rows('predictions.jsonl')}
inventory=[]
for qid,j in judgments.items():
 if j['status']=='judged' and j['correct']:continue
 r=retrievals.get(qid);answer=predictions.get(qid);coverage=(r or {}).get('source_coverage')
 if j['status']!='judged':label=j['status']
 elif not r or not r['memories']:label='empty_evidence'
 elif coverage and not coverage['any']:label='no_gold_source_id_recalled'
 elif coverage and not coverage['all']:label='partial_gold_source_ids_recalled'
 elif coverage:label='all_gold_source_ids_recalled_but_wrong'
 else:label='wrong_without_source_id_diagnostic'
 inventory.append({'qid':qid,'sample_id':j['sample_id'],'category':j['category'],'observed_bucket':label,'status':j['status'],'source_coverage':coverage,'evidence_count':len(r['memories']) if r else None,'answer':answer.get('answer') if answer else None,'judge_raw':j.get('raw'),'error':j.get('error')})
output=directory/'error-analysis';output.mkdir(exist_ok=True)
result={'run_id':a.run_id,'planned':manifest['planned_questions'],'wrong_judged':sum(j['status']=='judged' and not j['correct'] for j in judgments.values()),'unjudged':manifest['planned_questions']-sum(j['status']=='judged' for j in judgments.values()),'observed_buckets':dict(collections.Counter(r['observed_bucket'] for r in inventory)),'scope':'Observed pipeline status and original source-ID recall only. These labels do not establish semantic support, causal blame, or Judge correctness. No model or service calls.','input_hashes':{n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in ['manifest.json','judgments.jsonl','retrievals.jsonl','predictions.jsonl'] if (directory/n).exists()}}
(output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');(output/'questions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in inventory));print(json.dumps(result,ensure_ascii=False))
