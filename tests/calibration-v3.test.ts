import {test} from 'node:test';import assert from 'node:assert/strict';
import {answerSpans,criteriaForV3,validateAuditV3,auditCaseV3} from '../src/calibration-v3.js';
import {createServer} from 'node:http';
import type {AuditCase} from '../src/calibration.js';
const row:AuditCase={qid:'omission',question:'Which appointment is confirmed?',reference:'June 8 confirmed; April 24 unconfirmed.',answer:'**June 8** is confirmed.',rubric:{must_include:['June 8'],harmful_extra:['Omits that April 24 is still unconfirmed.']},evaluation_type:'OperationTrace'};
const output=()=>({criteria:criteriaForV3(row).map(c=>({id:c.id,requirement:'present',verdict:c.id==='reference'||c.id.startsWith('harmful_extra:')?'violated':'satisfied',answer_span_ids:c.id==='must_include:0'?[0]:[],reason:'The April candidate is omitted, while June is stated.'}))});
test('omission-type harmful extras fail without requiring a quote of absent information',()=>{
 const result=validateAuditV3(row,JSON.stringify(output()));assert.equal(result.status,'judged');assert.equal(result.correct,false);assert.deepEqual(result.criteria[2]!.answer_quotes,[]);
});
test('span references restore exact case and Markdown from the saved answer',()=>{
 const r={...row,answer:'You selected **June 8**.\nApril 24 remains unconfirmed.'};const spans=answerSpans(r.answer);
 assert.ok(spans.every(s=>r.answer.slice(s.start,s.end)===s.text));
 const raw={criteria:criteriaForV3(r).map(c=>({id:c.id,requirement:'present',verdict:'satisfied',answer_span_ids:c.id==='must_include:0'?[0]:[0,1],reason:'Both dates and their status are present.'}))};
 const result=validateAuditV3(r,JSON.stringify(raw));assert.equal(result.correct,true);assert.equal(result.criteria[1]!.answer_quotes[0],'You selected **June 8**.');
});
test('a prohibited assertion cannot be reclassified as an omission to avoid evidence',()=>{
 const r={...row,answer:'My PIN is ZX-482.',rubric:{must_not_include:['ZX-482']}};
 const raw={criteria:[{id:'reference',requirement:'present',verdict:'violated',answer_span_ids:[],reason:'Wrong response.'},{id:'must_not_include:0',requirement:'present',verdict:'violated',answer_span_ids:[],reason:'Misclassified.'}]};
 assert.throws(()=>validateAuditV3(r,JSON.stringify(raw)),/requirement/);
 raw.criteria[1]!.requirement='absent';assert.throws(()=>validateAuditV3(r,JSON.stringify(raw)),/span/);
 raw.criteria[1]!.answer_span_ids=[0];assert.equal(validateAuditV3(r,JSON.stringify(raw)).correct,false);
});
test('missing, duplicate and invalid span references are protocol errors, not passes',()=>{
 const missing=output();missing.criteria.pop();assert.throws(()=>validateAuditV3(row,JSON.stringify(missing)),/coverage/);
 const duplicate=output();duplicate.criteria.push(duplicate.criteria[0]!);assert.throws(()=>validateAuditV3(row,JSON.stringify(duplicate)),/duplicate/);
 for(const ids of [[90],[0,0],[-1],[0.5]]){const bad=output();bad.criteria[1]!.answer_span_ids=ids;assert.throws(()=>validateAuditV3(row,JSON.stringify(bad)),/span/);}
});
test('long Unicode spans preserve original UTF-16 positions and conflicting rubrics remain reviewable',()=>{
 const answer=('甲😀**bold**').repeat(150);const spans=answerSpans(answer);assert.equal(spans.map(s=>s.text).join(''),answer);assert.ok(spans.every(s=>s.text.length<=480&&!/[\uD800-\uDBFF]$/.test(s.text)));
 const raw=output();raw.criteria[2]!.verdict='uncertain';assert.equal(validateAuditV3(row,JSON.stringify(raw)).correct,null);
});
test('live audit input excludes control labels and preserves the exact saved answer',async()=>{
 let input:any;
 const server=createServer(async(req,res)=>{
  let body='';for await(const chunk of req)body+=chunk;
  input=JSON.parse(body);
  res.writeHead(200,{'content-type':'application/json'});
  res.end(JSON.stringify({choices:[{message:{content:JSON.stringify(output())}}]}));
 });
 await new Promise<void>(resolve=>server.listen(0,'127.0.0.1',resolve));
 try{
  const caseWithLabels={...row,expected_correct:false,expected_requirements:{'harmful_extra:0':'present'},expected_reason:'CONTROL_ONLY_LABEL'};
  const result=await auditCaseV3(caseWithLabels,{base:`http://127.0.0.1:${(server.address() as any).port}`,key:'fixture',model:'fixture'});
  assert.equal(result.status,'judged');assert.equal(result.correct,false);
  const sent=JSON.parse(input.messages[1].content);
  assert.equal(sent.ANSWER,row.answer);assert.deepEqual(sent.ANSWER_SPANS,answerSpans(row.answer));
  for(const key of ['expected_correct','expected_requirements','expected_reason'])assert.equal(key in sent,false);
  assert.ok(!JSON.stringify(input).includes('CONTROL_ONLY_LABEL'));
 }finally{server.closeAllConnections();await new Promise<void>(resolve=>server.close(()=>resolve()));}
});
