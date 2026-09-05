"""Paired comparisons from persisted evaluation artifacts; never calls a service."""
import argparse,collections,hashlib,json,pathlib,random,statistics,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--include-campaign',action='append',default=[]);p.add_argument('--output',required=True);a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
dataset_paths=sorted(set((root/'.data').glob('*-dev.json'))|set((root/'.data').glob('*-test.json')))
datasets={hashlib.sha256(path.read_bytes()).hexdigest():path for path in dataset_paths}
def grouped(manifest,baseline=None):
 data=datasets.get(manifest['dataset_sha256'])
 if data is None:return {'status':'unavailable','reason':'Exact dataset needed for group_id mapping is not present'}
 command=[sys.executable,str(root/'scripts/grouped-statistics.py'),'--run-id',manifest['run_id'],'--data',str(data)]
 if baseline:command+=['--compare-run',baseline]
 return json.loads(subprocess.check_output(command,text=True))
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
 item['input_hashes']={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in ['manifest.json','metrics.json','judgments.jsonl','retrievals.jsonl'] if (directory/name).exists()}
 item['dataset_group_statistics']=grouped(manifest)
 table.append(item);runs[(profile,benchmark)]=(manifest,judgments)
pairs=[]
edges={(profile,'U3',benchmark) for profile,benchmark in runs if profile!='U3' and ('U3',benchmark) in runs}
sequence=[('U0','U1'),('B0','B1'),('B1','B2'),('B2','B3'),('B3','B4'),('B4','B5'),('B5','B6')]
edges|={(before,after,benchmark) for before,after in sequence for benchmark in ['locomo','memops'] if (before,benchmark) in runs and (after,benchmark) in runs}
for profile,candidate,benchmark in sorted(edges):
 manifest,judgments=runs[(profile,benchmark)]
 target,target_rows=runs[(candidate,benchmark)];keys=['dataset_sha256','planned_questions','samples','answer_model','answer_inference','answer_base','judge_model','judge_kind','judge_base','answer_prompt_sha256','rubric_prompt_sha256','top_k','chunk_messages','chunk_words']
 mismatch=[key for key in keys if target.get(key)!=manifest.get(key)]
 if mismatch:pairs.append({'baseline':profile,'candidate':candidate,'benchmark':benchmark,'comparison':'refused','mismatched_fields':mismatch});continue
 common=set(judgments)&set(target_rows);changes=[];samples=collections.defaultdict(list)
 for qid in sorted(common):
  before=judgments[qid];after=target_rows[qid];old=int(before.get('correct',False) and before['status']=='judged');new=int(after.get('correct',False) and after['status']=='judged');samples[after['sample_id']].append(new-old)
  if old!=new:changes.append({'qid':qid,'direction':'wrong_to_correct' if new else 'correct_to_wrong','baseline_status':before['status'],'candidate_status':after['status']})
 rng=random.Random(20260905);groups=list(samples.values());bootstrap=[]
 for _ in range(1000):
  draw=[rng.choice(groups) for _ in groups];values=[d for group in draw for d in group];bootstrap.append(sum(values)/len(values) if values else 0)
 joint=[q for q in common if judgments[q]['status']=='judged' and target_rows[q]['status']=='judged']
 joint_before=sum(bool(judgments[q]['correct']) for q in joint);joint_after=sum(bool(target_rows[q]['correct']) for q in joint)
 kind='same_candidate_repeat' if (profile,candidate)==('B6','U3') else 'refactor_quality_repeat' if (profile,candidate)==('U0','U1') else 'component_sequence' if (profile,candidate) in sequence else 'full_candidate_comparison'
 pairs.append({'baseline':profile,'candidate':candidate,'benchmark':benchmark,'comparison_kind':kind,'paired_questions':len(common),'jointly_judged':len(joint),'joint_diagnostic':{'baseline_correct':joint_before,'candidate_correct':joint_after,'delta':(joint_after-joint_before)/len(joint) if joint else None,'scope':'Conditional subset only; excluded failures may bias this diagnostic.'},'wrong_to_correct':sum(x['direction']=='wrong_to_correct' for x in changes),'correct_to_wrong':sum(x['direction']=='correct_to_wrong' for x in changes),'delta_bootstrap_95':[quantile(bootstrap,.025),quantile(bootstrap,.975)],'changes':changes,'scope':'errors count as non-correct in this planned-denominator comparison; jointly_judged exposes infrastructure imbalance'})
 group_result=grouped(target,manifest['run_id'])
 pairs[-1]['dataset_group_comparison']={k:group_result[k] for k in ['status','reason','group_units','group_question_counts','paired','hashes','resampling'] if k in group_result}
result={'campaign':a.campaign,'included_campaigns':a.include_campaign,'comparison_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'runs':table,'paired_comparisons':pairs};(out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['# 开发集配对对照','',f'实验批次：`{a.campaign}`。以下为本地配置的代理结果，不是正式平台成绩。','', '| 配置 | 基准 | 正确/计划 | 已判定 | search p95 ms |','|---|---|---:|---:|---:|']
for item in table:lines.append(f"| {item['profile']} | {item['benchmark']} | {item['correct']}/{item['planned']} | {item['judged']} | {item['search_ms']['p95']} |")
lines+=['','配对变化、按 sample 重采样的区间、服务/评测版本、错误类型及证据成本见 `comparison.json`。包含相对U3的比较及B0→B1→…→B6的逐阶段配对。B6与U3是相同完整候选配置的重复运行，不能把两者差异当作组件收益。基础设施失败保留在计划分母中；共同完成判分的问题数量用于辨别运行失败与算法差异。两个 LoCoMo 开发组不足以支持稳定的泛化结论。','']
lines+=['若本地存在hash匹配的划分数据，dataset_group_statistics及dataset_group_comparison额外按group_id整体重采样，MemOps同一背景下的操作变体属于同一单位。原始sample区间保留并与背景区间分开；本次MemOps开发集仅3个背景，不能按30道独立题解释区间。','']
(out/'comparison.md').write_text('\n'.join(lines));print(json.dumps({'completed_runs':len(table),'paired_comparisons':len(pairs)}))
