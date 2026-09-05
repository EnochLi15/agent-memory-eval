"""Paired comparisons from persisted evaluation artifacts; never calls a service."""
import argparse,collections,json,pathlib,random,statistics
p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--include-campaign',action='append',default=[]);p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
def rows(path):return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
def quantile(values,q):
 values=sorted(values);return values[min(len(values)-1,int(len(values)*q))] if values else None
runs={};table=[]
campaigns=[a.campaign,*a.include_campaign]
directories=sorted({p for campaign in campaigns for p in (root/'artifacts').glob(campaign+'-*')})
for directory in directories:
 if not (directory/'metrics.json').exists():continue
 manifest=json.loads((directory/'manifest.json').read_text())
 if manifest['status']!='finished':continue
 metric=json.loads((directory/'metrics.json').read_text());judgments={r['qid']:r for r in rows(directory/'judgments.jsonl')};retrievals=rows(directory/'retrievals.jsonl')
 campaign=max((c for c in campaigns if directory.name.startswith(c+'-')),key=len)
 profile=directory.name[len(campaign)+1:].rsplit('-',1)[0];benchmark=directory.name.rsplit('-',1)[1]
 if (profile,benchmark) in runs:raise SystemExit('Ambiguous repeated profile/benchmark across campaigns; choose unique sources')
 item={'run_id':directory.name,'profile':profile,'benchmark':benchmark,'service_commit':manifest['service_commit'],'eval_commit':manifest['eval_commit'],'planned':metric['planned'],'judged':metric['judged'],'correct':metric['correct'],'accuracy_over_planned':metric['accuracy_over_planned'],'sample_bootstrap_95':metric['sample_bootstrap_95'],'add_ms':metric['add_ms'],'search_ms':metric['search_ms'],'error_counts':dict(collections.Counter(r['status'] for r in judgments.values() if r['status']!='judged')),'evidence':{'queries':len(retrievals),'mean_count':statistics.mean(len(r['memories']) for r in retrievals) if retrievals else None,'mean_utf8_bytes':statistics.mean(sum(len(m['content'].encode()) for m in r['memories']) for r in retrievals) if retrievals else None,'source_coverage':metric['evidence_coverage']}}
 table.append(item);runs[(profile,benchmark)]=(manifest,judgments)
pairs=[]
for (profile,benchmark),(manifest,judgments) in runs.items():
 if profile=='U3' or ('U3',benchmark) not in runs:continue
 target,target_rows=runs[('U3',benchmark)];keys=['dataset_sha256','answer_model','judge_model','judge_kind','answer_prompt_sha256','rubric_prompt_sha256','top_k','chunk_messages','chunk_words']
 mismatch=[key for key in keys if target.get(key)!=manifest.get(key)]
 if mismatch:pairs.append({'profile':profile,'benchmark':benchmark,'comparison':'refused','mismatched_fields':mismatch});continue
 common=set(judgments)&set(target_rows);changes=[];samples=collections.defaultdict(list)
 for qid in sorted(common):
  before=judgments[qid];after=target_rows[qid];old=int(before.get('correct',False) and before['status']=='judged');new=int(after.get('correct',False) and after['status']=='judged');samples[after['sample_id']].append(new-old)
  if old!=new:changes.append({'qid':qid,'direction':'wrong_to_correct' if new else 'correct_to_wrong','baseline_status':before['status'],'candidate_status':after['status']})
 rng=random.Random(20260905);groups=list(samples.values());bootstrap=[]
 for _ in range(1000):
  draw=[rng.choice(groups) for _ in groups];values=[d for group in draw for d in group];bootstrap.append(sum(values)/len(values) if values else 0)
 joint=[q for q in common if judgments[q]['status']=='judged' and target_rows[q]['status']=='judged']
 joint_before=sum(bool(judgments[q]['correct']) for q in joint);joint_after=sum(bool(target_rows[q]['correct']) for q in joint)
 pairs.append({'baseline':profile,'candidate':'U3','benchmark':benchmark,'paired_questions':len(common),'jointly_judged':len(joint),'joint_diagnostic':{'baseline_correct':joint_before,'candidate_correct':joint_after,'delta':(joint_after-joint_before)/len(joint) if joint else None,'scope':'Conditional subset only; excluded failures may bias this diagnostic.'},'wrong_to_correct':sum(x['direction']=='wrong_to_correct' for x in changes),'correct_to_wrong':sum(x['direction']=='correct_to_wrong' for x in changes),'delta_bootstrap_95':[quantile(bootstrap,.025),quantile(bootstrap,.975)],'changes':changes,'scope':'errors count as non-correct in this planned-denominator comparison; jointly_judged exposes infrastructure imbalance'})
result={'campaign':a.campaign,'included_campaigns':a.include_campaign,'runs':table,'paired_comparisons':pairs};(out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['# 开发集配对对照','',f'实验批次：`{a.campaign}`。以下为本地配置的代理结果，不是正式平台成绩。','', '| 配置 | 基准 | 正确/计划 | 已判定 | search p95 ms |','|---|---|---:|---:|---:|']
for item in table:lines.append(f"| {item['profile']} | {item['benchmark']} | {item['correct']}/{item['planned']} | {item['judged']} | {item['search_ms']['p95']} |")
lines+=['','配对变化、按 sample 重采样的区间、服务/评测版本、错误类型及证据成本见 `comparison.json`。基础设施失败保留在计划分母中；共同完成判分的问题数量用于辨别运行失败与算法差异。两个 LoCoMo 开发组不足以支持稳定的泛化结论。','']
(out/'comparison.md').write_text('\n'.join(lines));print(json.dumps({'completed_runs':len(table),'paired_comparisons':len(pairs)}))
