"""Post-hoc upstream MemOps judge, separate from competition-style rubric score.

Reads saved predictions and pinned public gold only. Never calls the memory
service or Answer model. Official prompt/parser/postchecks remain unmodified.
"""
import argparse,hashlib,importlib.util,json,os,pathlib,re,subprocess,sys,time
from openai import OpenAI
root=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--model',default='gpt-5.4-mini');p.add_argument('--limit',type=int,default=0);args=p.parse_args()
if not re.fullmatch(r'[A-Za-z0-9_.-]+',args.run_id):raise SystemExit('Invalid run id')
directory=root/'artifacts'/args.run_id;output=directory/'upstream-memops-diagnostics'
if output.exists():raise SystemExit('Diagnostic output exists; preserve the previous result')
output.mkdir()
upstream=root/'python/upstream/memops/operation_metrics.py';spec=importlib.util.spec_from_file_location('memops_metrics',upstream);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
client=OpenAI(base_url=os.environ['MEMORY_LLM_BASE_URL'],api_key=os.environ['MEMORY_LLM_API_KEY'],max_retries=2,timeout=90)
def caller(prompt,model):
 start=time.monotonic();result='';finish=None;usage=None
 for chunk in client.chat.completions.create(model=model,messages=[{'role':'user','content':prompt}],stream=True,stream_options={'include_usage':True},max_completion_tokens=1800,response_format={'type':'json_object'}):
  if chunk.choices:
   result+=chunk.choices[0].delta.content or '';finish=chunk.choices[0].finish_reason or finish
  if chunk.usage:usage=chunk.usage.model_dump()
 with (output/'calls.jsonl').open('a') as f:f.write(json.dumps({'elapsed_ms':(time.monotonic()-start)*1000,'usage':usage,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})+'\n')
 if finish!='stop':raise ValueError('Incomplete judge response')
 return result
predictions=[json.loads(l) for l in (directory/'predictions.jsonl').read_text().splitlines()]
selected=[r for r in predictions if r['benchmark']=='memops'];selected=selected[:args.limit] if args.limit else selected
revision='312af65e2c7b6d1b70f062ffa8b4cde32aaf6f35';cache={};results=[];hashes={}
for row in selected:
 try:
  parts=row['qid'].split('#');name=parts[0];ordinal=int(parts[-1])
  if not re.fullmatch(r'[A-Za-z0-9_]+\.json',name):raise ValueError('Invalid source filename')
  if name not in cache:
   full=json.loads((root/'.data/MemOps/generated_result/4-inject_evidence_with_distractors'/name).read_text())
   data=subprocess.check_output(['git','show',revision+':generated_result/2-evidence_conversation/'+name],cwd=root/'.data/MemOps')
   gold=json.loads(data);hashes[name]=hashlib.sha256(data).hexdigest();(output/name).write_bytes(data);cache[name]=(full,gold)
  full,gold=cache[name];question=full['answer'][ordinal]
  entry={**question,'question_id':row['qid'],'source_file':name,'operation_type':full['operation_type'],'gold_operations':gold.get('operations',[]),'hypothesis':row['answer'],'evaluation_method':'http-memory-evidence','model':args.model}
  result=mod.evaluate_entry(entry,judge_model=args.model,call_llm=caller,evidence_dirs=[output]);result['status']='judged'
 except Exception as error:result={'question_id':row['qid'],'status':'error','error_type':type(error).__name__}
 results.append(result)
 with (output/'results.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
 print(json.dumps({'completed':len(results),'planned':len(selected),'status':result['status']}),flush=True)
judged=[r for r in results if r['status']=='judged'];summary={'scope':'upstream MemOps diagnostics on saved answers, not competition score; generation model differs from upstream default','planned':len(selected),'judged':len(judged),'answer_correct':sum(r.get('answer_score',0) for r in judged),'lifecycle':mod.lifecycle_metric_summary(judged)}
(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(output/'manifest.json').write_text(json.dumps({'upstream_revision':revision,'script_sha256':hashlib.sha256(upstream.read_bytes()).hexdigest(),'gold_files':hashes,'model':args.model,'source_run':args.run_id,'prediction_sha256':hashlib.sha256((directory/'predictions.jsonl').read_bytes()).hexdigest()},indent=2)+'\n')
