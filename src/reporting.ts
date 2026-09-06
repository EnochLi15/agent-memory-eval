/** Descriptive resampling of recorded outcomes, never an estimate of the
 * unknown correctness of questions that could not be answered or judged. */
export function groupOutcomeSummary(manifest:any,judgments:any[]){
 const groups=new Map<string,{benchmark:string;group_id:string;recorded:number;correct:number}>();let metadataComplete=true;
 for(const j of judgments){
  const identity=manifest.sample_groups?.[j.sample_id];
  if(!identity||identity.benchmark!==j.benchmark||typeof identity.group_id!=='string'||!identity.group_id){metadataComplete=false;continue;}
  const key=JSON.stringify([identity.benchmark,identity.group_id]);let group=groups.get(key);
  if(!group){group={benchmark:identity.benchmark,group_id:identity.group_id,recorded:0,correct:0};groups.set(key,group);}group!.recorded++;if(j.status==='judged'&&j.correct)group!.correct++;
 }
 const totals=[...groups.values()],complete=judgments.length===manifest.planned_questions;
 let interval:[number,number]|null=null;
 if(metadataComplete&&complete&&totals.length>=2){
  let seed=20260905;const random=()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed/4294967296;};
  const values=Array.from({length:1000},()=>{let n=0,c=0;for(let i=0;i<totals.length;i++){const g=totals[Math.floor(random()*totals.length)]!;n+=g.recorded;c+=g.correct;}return c/n;}).sort((a,b)=>a-b);
  interval=[values[25]!,values[975]!];
 }
 return {protocol:'recorded-group-outcome-bootstrap-v1',group_count:metadataComplete?totals.length:null,groups:metadataComplete?totals:[],interval_95:interval,unavailable_reason:!metadataComplete?'missing_group_metadata':!complete?'unrecorded_questions':totals.length<2?'fewer_than_two_groups':null,scope:'Resample declared conversation/background groups with all variants together. Infrastructure failures remain in the denominator; this does not infer unjudged answer quality or unseen-data performance.'};
}
