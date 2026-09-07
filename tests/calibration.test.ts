import {test} from 'node:test';
import assert from 'node:assert/strict';
import {criteriaFor,validateAudit,type AuditCase} from '../src/calibration.js';
import {completion} from '../src/models.js';
const row:AuditCase={qid:'x',question:'Current setup?',answer:'Current tier: 300 GB. Previously 100 GB.',reference:'300 GB',rubric:{must_include:['300 GB'],must_not_include:['100 GB as current tier']},evaluation_type:'StateTransition'};
const verdict=()=>({criteria:criteriaFor(row).map(c=>({id:c.id,verdict:'satisfied',answer_quotes:c.kind==='required'?['300 GB']:[],reason:'Criterion satisfied.'}))});
test('criterion aggregation cannot pass with a missing fact even if model supplies an overall pass',()=>{
 const v=verdict();v.criteria[1]!.verdict='violated';v.criteria[1]!.answer_quotes=[];v.criteria[1]!.reason='Required current tier is absent.';
 const r=validateAudit(row,JSON.stringify({...v,correct:true}));assert.equal(r.correct,false);assert.deepEqual(r.consistency_flags,['reference_satisfied_but_rubric_violated']);
});
test('larger audit budget does not change the ordinary Answer default',async()=>{
 const previous=globalThis.fetch;const bodies:any[]=[];
 globalThis.fetch=async(_url,init)=>{bodies.push(JSON.parse(String(init?.body)));return new Response(JSON.stringify({choices:[{message:{content:'OK'},finish_reason:'stop'}]}));};
 try{
  await completion('http://fixture.invalid/v1','fixture','fixture',[]);
  await completion('http://fixture.invalid/v1','fixture','fixture',[],true,6000);
  assert.equal(bodies[0].max_completion_tokens,1800);assert.equal(bodies[1].max_completion_tokens,6000);assert.equal(bodies[0].stream,true);
 }finally{globalThis.fetch=previous;}
});
test('incomplete, duplicated and reference-only quotations are not accepted as successful audits',()=>{
 const v=verdict();v.criteria.pop();assert.throws(()=>validateAudit(row,JSON.stringify(v)),/coverage/);
 const duplicate=verdict();duplicate.criteria.push(duplicate.criteria[0]!);assert.throws(()=>validateAudit(row,JSON.stringify(duplicate)),/duplicate/);
 const invented=verdict();invented.criteria[0]!.answer_quotes=['This quote was never in the answer'];assert.throws(()=>validateAudit(row,JSON.stringify(invented)),/Quote/);
});
test('uncertain rubric needs review and literal diagnostic does not silently decide semantics',()=>{
 const v=verdict();v.criteria[2]!.verdict='uncertain';assert.equal(validateAudit(row,JSON.stringify(v)).correct,null);
 const r={...row,rubric:{must_not_include:['100 GB']}};const raw={criteria:criteriaFor(r).map(c=>({id:c.id,verdict:'satisfied',answer_quotes:c.kind==='required'?['300 GB']:[],reason:'Prior tier is permitted historical context.'}))};
 const result=validateAudit(r,JSON.stringify(raw));assert.equal(result.correct,true);assert.equal(result.literal_diagnostics.matches.length,1);
});
