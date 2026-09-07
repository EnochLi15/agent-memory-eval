import {readFileSync,writeFileSync,appendFileSync,mkdirSync,existsSync} from 'node:fs';
import {resolve,join} from 'node:path';
import {createHash} from 'node:crypto';
import {parseEnv} from 'node:util';
import {auditCase,CALIBRATION_PROTOCOL,CALIBRATION_PROMPT} from '../dist/calibration.js';
import {auditCaseV3,CALIBRATION_PROTOCOL_V3,CALIBRATION_PROMPT_V3} from '../dist/calibration-v3.js';

const args=process.argv.slice(2);const opt=k=>args[args.indexOf('--'+k)+1];
const version=args.includes('--protocol')?opt('protocol'):'v2';if(!['v2','v3'].includes(version))throw Error('Expected --protocol v2 or v3');
const audit=version==='v3'?auditCaseV3:auditCase,protocol=version==='v3'?CALIBRATION_PROTOCOL_V3:CALIBRATION_PROTOCOL,prompt=version==='v3'?CALIBRATION_PROMPT_V3:CALIBRATION_PROMPT;
for(const k of ['data','output'])if(!args.includes('--'+k)||!opt(k))throw Error('Required --'+k);
const input=resolve(opt('data')),output=resolve(opt('output'));if(existsSync(output))throw Error('Output exists; choose a new audit run');
const bytes=readFileSync(input);const parsed=input.endsWith('.jsonl')?bytes.toString().split('\n').filter(Boolean).map(l=>JSON.parse(l)):JSON.parse(bytes);
const cases=Array.isArray(parsed)?parsed:parsed.cases;if(!Array.isArray(cases)||!cases.length)throw Error('No audit cases');
if(new Set(cases.map(c=>c.qid)).size!==cases.length)throw Error('Duplicate audit identity');
const env={...parseEnv(readFileSync(args.includes('--env')?resolve(opt('env')):resolve('../.env'),'utf8')),...process.env};
const connection={base:env.EVALUATOR_API_BASE??env.MEMORY_LLM_BASE_URL,key:env.EVALUATOR_API_KEY??env.MEMORY_LLM_API_KEY,model:env.EVALUATOR_MODEL??env.MEMORY_LLM_MODEL};
if(!connection.base||!connection.key||!connection.model)throw Error('Missing evaluator connection');
const sha=s=>createHash('sha256').update(s).digest('hex');
mkdirSync(output,{recursive:true});
const manifest={protocol,input_sha256:sha(bytes),input_provenance:Array.isArray(parsed)?null:parsed.provenance??null,prompt_sha256:sha(prompt),source_sha256:sha(readFileSync(new URL(version==='v3'?'../src/calibration-v3.ts':'../src/calibration.ts',import.meta.url))),...(version==='v3'?{shared_semantics_sha256:sha(readFileSync(new URL('../src/calibration.ts',import.meta.url)))}:{}),model_runtime_sha256:sha(readFileSync(new URL('../src/models.ts',import.meta.url))),runner_sha256:sha(readFileSync(new URL(import.meta.url))),model:connection.model,base:connection.base,inference:{stream:true,max_completion_tokens:6000,transport_attempts:3,semantic_retries:0},planned:cases.length,concurrency:2,max_consecutive_transport_failures:4,started_at:new Date().toISOString(),status:'running',scope:'Posthoc model criterion audit. Neither independent human ground truth nor a replacement for historical benchmark scores.'};
writeFileSync(join(output,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
let cursor=0,consecutiveFailures=0,paused=false;const rows=[];
await Promise.all(Array.from({length:2},async()=>{
 while(cursor<cases.length&&!paused){const c=cases[cursor++];const start=performance.now();let row;
  try{row={qid:c.qid,evaluation_type:c.evaluation_type,...await audit(c,connection)};}
  catch(error){row={qid:c.qid,evaluation_type:c.evaluation_type,status:'error',correct:null,error:error instanceof Error?error.message:'Unknown audit error'};}
  row.elapsed_ms=performance.now()-start;
  consecutiveFailures=row.status==='error'?consecutiveFailures+1:0;
  if(consecutiveFailures>=manifest.max_consecutive_transport_failures)paused=true;
  if(typeof c.expected_correct==='boolean'){row.control_expected=c.expected_correct;row.control_match=row.status==='judged'&&row.correct===c.expected_correct;}
  rows.push(row);appendFileSync(join(output,'results.jsonl'),JSON.stringify(row)+'\n');
  console.log(JSON.stringify({completed:rows.length,planned:cases.length,qid:c.qid,status:row.status,correct:row.correct,...('control_match'in row?{control_match:row.control_match}:{})}));
 }
}));
for(const c of cases.slice(cursor)){
 const row={qid:c.qid,evaluation_type:c.evaluation_type,status:'not_attempted',correct:null,reason:'Circuit opened after repeated transport failures'};
 rows.push(row);appendFileSync(join(output,'results.jsonl'),JSON.stringify(row)+'\n');
}
const count=key=>Object.fromEntries([...new Set(rows.map(r=>r[key]))].map(v=>[v,rows.filter(r=>r[key]===v).length]));
const controls=rows.filter(r=>'control_expected'in r);
const summary={protocol,planned:cases.length,status_counts:count('status'),correct:rows.filter(r=>r.correct===true).length,incorrect:rows.filter(r=>r.correct===false).length,controls:controls.length,control_matches:controls.filter(r=>r.control_match).length,by_type:Object.fromEntries([...new Set(rows.map(r=>r.evaluation_type))].map(type=>[type,{planned:rows.filter(r=>r.evaluation_type===type).length,judged:rows.filter(r=>r.evaluation_type===type&&r.status==='judged').length,control_matches:rows.filter(r=>r.evaluation_type===type&&r.control_match).length}])),scope:manifest.scope};
writeFileSync(join(output,'summary.json'),JSON.stringify(summary,null,2)+'\n');
writeFileSync(join(output,'manifest.json'),JSON.stringify({...manifest,status:rows.every(r=>r.status==='judged')?'finished':'incomplete',finished_at:new Date().toISOString(),results_sha256:sha(readFileSync(join(output,'results.jsonl')))},null,2)+'\n');
console.log(JSON.stringify(summary));
