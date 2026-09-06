import {completion} from './models.js';
import {criteriaFor,CALIBRATION_PROMPT,type AuditCase} from './calibration.js';
export const CALIBRATION_PROTOCOL_V3='semantic-criteria-span-audit-v3';
export type AnswerSpan={id:number;start:number;end:number;text:string};
type Requirement='present'|'absent';
type Result={id:string;requirement:Requirement;verdict:'satisfied'|'violated'|'uncertain';answer_span_ids:number[];answer_quotes:string[];reason:string};

/** IDs address the exact saved answer; the model never has to retype Markdown
 * or change its case. Long lines are bounded without splitting a surrogate. */
export function answerSpans(answer:string):AnswerSpan[]{
 const spans:AnswerSpan[]=[];
 for(const line of answer.matchAll(/[^\r\n]+/g)){
  let start=line.index!;const limit=start+line[0].length;
  while(start<limit){let end=Math.min(limit,start+480);if(end<limit&&/[\uD800-\uDBFF]/.test(answer[end-1]!))end--;
   const text=answer.slice(start,end);if(text.trim())spans.push({id:spans.length,start,end,text});start=end;
  }
 }
 return spans;
}
export function criteriaForV3(row:AuditCase){
 return criteriaFor(row).map(c=>({id:c.id,rule:c.rule,allowed_requirements:(c.id.startsWith('harmful_extra:')?['present','absent']:c.kind==='required'?['present']:['absent']) as Requirement[]}));
}
export const CALIBRATION_PROMPT_V3=CALIBRATION_PROMPT.split('Return JSON')[0]+`
For this v3 protocol, interpret the exact supplied criteria without inventing new rules. Each criterion has allowed_requirements:
- present means the criterion requires information in the answer. It is violated when that required meaning is missing or contradicted.
- absent means the criterion forbids a meaning in the answer. It is violated when that forbidden meaning occurs.
Reference completeness and must_include require present. must_not_include requires absent. A harmful_extra rule may describe either a forbidden assertion (absent) or a harmful omission (present). For example, 'Omits that the proposed appointment is unconfirmed' requires that status to be present; failing it does not require quoting nonexistent text. Do not reinterpret a plain prohibited assertion as an omission. Explain the chosen interpretation in the reason. If wording is contradictory or cannot be reconciled with the question/reference, give uncertain.
Return JSON {"criteria":[{"id":"...","requirement":"present|absent","verdict":"satisfied|violated|uncertain","answer_span_ids":[0],"reason":"brief specific explanation"}]}. Include every criterion once. Use only the exact IDs in ANSWER_SPANS, which are excerpts of the saved ANSWER. Do not output copied quotations, character offsets, replacements or an overall score. Cite spans for satisfied present requirements and violated absent requirements. Missing required meanings may use empty span IDs; explain what is missing. Satisfied absent requirements use empty span IDs because no excerpt proves absence. A span proves only what its actual text says; do not infer a missing date, actor or qualification merely because you can cite related words. Preserve the full question/reference/rubric when reasoning; span boundaries are citation mechanics, not semantic boundaries.`;

export function validateAuditV3(row:AuditCase,raw:string){
 const out=JSON.parse(raw) as {criteria?:unknown};if(!Array.isArray(out.criteria))throw Error('Missing criterion verdicts');
 const expected=criteriaForV3(row),byId=new Map(expected.map(c=>[c.id,c])),spans=answerSpans(row.answer),seen=new Set<string>(),results:Result[]=[];
 for(const c of out.criteria as any[]){
  const rule=c&&byId.get(c.id);if(!rule||seen.has(c.id))throw Error('Unknown or duplicate criterion id');seen.add(c.id);
  if(!rule.allowed_requirements.includes(c.requirement))throw Error('Invalid criterion requirement interpretation');
  if(!['satisfied','violated','uncertain'].includes(c.verdict)||!Array.isArray(c.answer_span_ids)||typeof c.reason!=='string'||!c.reason.trim())throw Error('Malformed criterion verdict');
  const ids=c.answer_span_ids as number[];if(new Set(ids).size!==ids.length||ids.some(id=>!Number.isInteger(id)||!spans[id]))throw Error('Invalid or duplicate answer span');
  const needsWitness=c.requirement==='present'&&c.verdict==='satisfied'||c.requirement==='absent'&&c.verdict==='violated';
  if(needsWitness&&!ids.length)throw Error('Criterion requires an answer span');
  if(c.requirement==='absent'&&c.verdict==='satisfied'&&ids.length)throw Error('An absence verdict cannot cite present text as proof of absence');
  results.push({id:c.id,requirement:c.requirement,verdict:c.verdict,answer_span_ids:ids,answer_quotes:ids.map(id=>spans[id]!.text),reason:c.reason});
 }
 if(seen.size!==expected.length)throw Error('Incomplete criterion coverage');
 const uncertain=results.some(c=>c.verdict==='uncertain'),flags=results.find(c=>c.id==='reference')?.verdict==='satisfied'&&results.some(c=>c.id!=='reference'&&c.verdict==='violated')?['reference_satisfied_but_rubric_violated']:[];
 const matches=expected.filter(c=>c.id.startsWith('must_not_include:')&&row.answer.toLowerCase().includes(c.rule.toLowerCase())).map(c=>({criterion:c.id,literal:c.rule}));
 return {status:uncertain?'needs_review' as const:'judged' as const,correct:uncertain?null:results.every(c=>c.verdict==='satisfied'),criteria:results,answer_spans:spans,consistency_flags:flags,literal_diagnostics:{scope:'Literal substring diagnostic only; does not decide semantic correctness.',matches}};
}
export async function auditCaseV3(row:AuditCase,connection:{base:string;key:string;model:string}){
 const input={question:row.question,options:row.options,reference:row.reference,rubric:row.rubric,ANSWER:row.answer,ANSWER_SPANS:answerSpans(row.answer),criteria:criteriaForV3(row)};
 const raw=await completion(connection.base,connection.key,connection.model,[{role:'system',content:CALIBRATION_PROMPT_V3},{role:'user',content:JSON.stringify(input)}],true,6000);
 try{return {raw,...validateAuditV3(row,raw)};}catch(error){return {raw,status:'protocol_error' as const,correct:null,error:error instanceof Error?error.message:'Invalid v3 audit response'};}
}
