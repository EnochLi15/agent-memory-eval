import {contract} from './contract.js';
import {resolve} from 'node:path';import {locomo,memops,competition,saveDataset} from './adapters.js';import {run,report} from './runner.js';import type {EvalConfig} from './types.js';
const args=process.argv.slice(2);const command=args.shift();const option=(name:string,fallback=''):string=>{const i=args.indexOf(`--${name}`);return i<0?fallback:args[i+1]??fallback;};
if(command==='contract'){await contract(option('base-url','http://127.0.0.1:8088'));}
else if(command==='prepare'){
  const kind=option('benchmark');let data;
  if(kind==='locomo')data=locomo(option('conversations'),option('questions'));
  else if(kind==='memops')data=memops(option('directory'),Number(option('files','0')));
  else if(kind==='competition')data=competition(option('input'));else throw new Error('Choose --benchmark locomo|memops|competition');
  saveDataset(data,option('output'));console.log(JSON.stringify({samples:data.length,questions:data.reduce((n,s)=>n+s.questions.length,0)}));
}else if(command==='run'){
  const e=process.env;const runId=option('run-id',`run-${Date.now()}`);const base=e.MEMORY_LLM_BASE_URL??'';const key=e.MEMORY_LLM_API_KEY??'';
  const config:EvalConfig={baseUrl:option('base-url','http://127.0.0.1:8088'),runDir:resolve(option('output',`artifacts/${runId}`)),runId,dataFile:resolve(option('data')),limit:Number(option('limit','0')),concurrency:Number(option('concurrency','2')),mode:option('mode','proxy') as EvalConfig['mode'],answerModel:option('answer-model',e.MEMORY_LLM_MODEL??'gpt-5.4-mini'),judgeModel:option('judge-model',e.MEMORY_LLM_MODEL??'gpt-5.4-mini'),llmBase:base,llmKey:key,judgeBase:e.EVALUATOR_API_BASE??base,judgeKey:e.EVALUATOR_API_KEY??key,judgeKind:option('judge-kind','rubric') as EvalConfig['judgeKind'],maxMessages:Number(option('chunk-messages','20')),maxWords:Number(option('chunk-words','2000')),topK:100,resume:args.includes('--resume'),memoryNamespace:option('memory-namespace',runId)};
  await run(config);
}else if(command==='report')report(resolve(option('directory')));
else console.log('Commands: prepare --benchmark ...; run --data ... --run-id ...; report --directory ...');
