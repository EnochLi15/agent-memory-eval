import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {spawn} from 'node:child_process';
import {mkdtempSync,writeFileSync,readFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

test('calibration audit stops scheduling after a transport outage and retains unattempted cases in the denominator',{timeout:20000},async()=>{
 const dir=mkdtempSync(join(tmpdir(),'audit-circuit-'));let calls=0;
 const server=createServer((_req,res)=>{calls++;res.writeHead(503,{'content-type':'application/json'});res.end('{"error":"fixture outage"}');});
 await new Promise<void>(resolve=>server.listen(0,'127.0.0.1',resolve));
 try{
  const base='http://127.0.0.1:'+(server.address() as any).port;
  const data=join(dir,'data.json'),envFile=join(dir,'fixture.env'),output=join(dir,'output');
  writeFileSync(envFile,`MEMORY_LLM_BASE_URL=${base}\nMEMORY_LLM_API_KEY=fixture-key\nMEMORY_LLM_MODEL=fixture\n`);
  writeFileSync(data,JSON.stringify(Array.from({length:7},(_,i)=>({qid:String(i),question:'Current city?',reference:'Oslo',rubric:{must_include:['Oslo']},answer:'Oslo',evaluation_type:'StateTransition'}))));
  const code=await new Promise<number|null>((resolve,reject)=>{
   const child=spawn(process.execPath,[new URL('../scripts/audit-calibration.mjs',import.meta.url).pathname,'--data',data,'--output',output,'--env',envFile],{env:{...process.env,EVALUATOR_API_BASE:base,EVALUATOR_API_KEY:'fixture-key',EVALUATOR_MODEL:'fixture'},stdio:'ignore'});
   child.on('error',reject);child.on('exit',resolve);
  });
  assert.equal(code,0);
  const rows=readFileSync(join(output,'results.jsonl'),'utf8').trim().split('\n').map(l=>JSON.parse(l));
  assert.equal(rows.length,7);const attempted=rows.filter(r=>r.status==='error').length;
  assert.ok(attempted>=4&&attempted<=5);assert.equal(calls,attempted*3);
  assert.equal(rows.filter(r=>r.status==='not_attempted').length,7-attempted);
  const manifest=JSON.parse(readFileSync(join(output,'manifest.json'),'utf8'));assert.equal(manifest.planned,7);assert.equal(manifest.status,'incomplete');
 }finally{server.closeAllConnections();await new Promise<void>(resolve=>server.close(()=>resolve()));rmSync(dir,{recursive:true,force:true});}
});
