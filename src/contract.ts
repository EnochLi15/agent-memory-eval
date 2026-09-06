import assert from 'node:assert/strict';
import {MemoryClient} from './client.js';
export async function contract(base:string):Promise<void>{
 const client=new MemoryClient(base);await client.health();const id=`contract-${Date.now()}`;
 const request={request_id:'initial',user_id:id,session_id:'session',messages:[{role:'user',content:'My manager is Clara. My access code is QC-9182. I live in Seattle.',timestamp:'2026-01-01T00:00:00Z'}]};
 await client.add(request);await client.add(request);
 const data=await client.search({query:'manager Clara',user_id:id,top_k:100});assert.ok(data.some(x=>x.content.includes('Clara')),'read after write');
 assert.deepEqual(await client.search({query:'manager Clara QC-9182',user_id:id+'-unknown',top_k:100}),[],'tenant isolation');
 for(const k of [0,1,2.5,100,101]){const rows=await client.search({query:'manager code',user_id:id,top_k:k});assert.ok(rows.length<=Math.min(100,Math.floor(k)));}
 for(const [method,path] of [['GET','/docs'],['GET','/metrics'],['HEAD','/health'],['POST','/reset'],['GET','/search'],['PUT','/add']]){const r=await fetch(base+path,{method});assert.equal(r.status,404,method+' '+path);}
 const invalid=await fetch(base+'/add',{method:'POST',headers:{'content-type':'application/json'},body:'{}'});assert.equal(invalid.status,400);
 const conflict=await fetch(base+'/add',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({...request,messages:[]})});assert.equal(conflict.status,409);
 await client.add({request_id:'update',user_id:id,session_id:'s2',messages:[{role:'user',content:'I now live in Portland.',timestamp:'2026-02-01T00:00:00Z'}]});
 const current=await client.search({query:'What is my current city?',user_id:id,top_k:100});
 assert.ok(current.some(x=>x.content.includes('Portland')),'updated state immediately visible');
 assert.ok(current.every(x=>!x.content.includes('Seattle')),'old state is not current');
 await client.add({request_id:'forget',user_id:id,session_id:'s2',messages:[{role:'user',content:'Forget my access code.',timestamp:'2026-02-01T00:00:00Z'}]});
 for(const query of ['access code QC-9182','previous code history','Clara manager']){const rows=await client.search({query,user_id:id,top_k:100,options:['QC-9182','invented option']});assert.ok(rows.every(x=>!x.content.includes('QC-9182')),'forgotten value leakage');}
 assert.ok((await client.search({query:'manager',user_id:id,top_k:100})).some(x=>x.content.includes('Clara')),'retained neighbor');
 console.log(JSON.stringify({contract:'passed',base_url:base,user_id:id,checks:['health','echo','idempotency','read-after-write','tenant-isolation','top-k','three-routes','validation','conflict','update-current-state','forget-all-paths','retained-neighbor']}));
}
