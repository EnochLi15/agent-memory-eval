import {readFileSync,readdirSync,writeFileSync,mkdirSync} from 'node:fs';
import {join,dirname} from 'node:path';
import type {Sample,Message} from './types.js';
type Obj=Record<string,any>;
const readLines=(path:string):Obj[]=>readFileSync(path,'utf8').split('\n').filter(Boolean).map(l=>JSON.parse(l) as Obj);

function sessionDate(raw:string,index:number):string{
  const rearranged=raw.replace(/(\d{1,2}:\d{2})\s*(am|pm) on (.+)/i,'$3 $1 $2');
  const ms=Date.parse(rearranged+' UTC');if(Number.isFinite(ms))return new Date(ms).toISOString();
  return new Date(Date.UTC(2025,0,1)+index*86400000).toISOString();
}
export function locomo(conversations:string,questions:string):Sample[]{
  const qs=readLines(questions);
  return readLines(conversations).map(c=>({sample_id:String(c.sample_id),group_id:String(c.sample_id),benchmark:'locomo',
    sessions:c.sessions.map((s:Obj,si:number)=>{
      const anchor=sessionDate(String(s.date_time),si);
      return {session_id:`session-${s.session_index}`,messages:s.messages.map((m:Obj,i:number)=>({role:String(m.role),content:`${i===0?`[Session time: ${s.date_time}]\n`:''}${m.speaker}: ${m.text}${m.blip_caption?`\n[Image description supplied with the conversation: ${m.blip_caption}]`:''}\n[Source id: ${m.dia_id}]`,timestamp:anchor}))};
    }),
    questions:qs.filter(q=>q.sample_id===c.sample_id).map(q=>({qid:String(q.qa_id),question:String(q.question),gold_answer:q.answer,category:String(q.category),gold_evidence:q.evidence??[]}))
  }));
}
export function memops(directory:string,filesLimit=0):Sample[]{
  const names=readdirSync(directory).filter(n=>n.endsWith('.json')).sort();
  return names.slice(0,filesLimit||names.length).map(name=>{
    const d=JSON.parse(readFileSync(join(directory,name),'utf8')) as Obj;
    return {sample_id:name.replace('.json',''),group_id:name.split('_')[0]!,benchmark:'memops',
      sessions:(d.conversations as Obj[]).map((s,i)=>({session_id:`segment-${s.segment_index??i}`,messages:(s.dialogue as Obj[]).map((m,j)=>({role:String(m.role),content:`${j===0?'[Session time: synthetic ordering only; use dates stated by speakers]\n':''}${String(m.content)}`,timestamp:new Date(Date.UTC(2025,0,1)+i*86400000+j*1000).toISOString()}))})),
      // Pair IDs intentionally repeat across evaluation settings in upstream.
      // Preserve each actual question as a distinct scored item.
      questions:(d.answer as Obj[]).map((q,i)=>({qid:`${name}#${q.question_pair_id??'question'}#${q.evaluation_setting??'unspecified'}#${i}`,question:String(q.question),gold_answer:q.expected_answer??'',gold_rubric:q.judge_rubric??{},options:q.options??q.candidate_options,category:`${d.operation_type}/${q.evaluation_type??'unknown'}`,gold_evidence:[]}))
    };
  });
}
export function competition(path:string):Sample[]{
 const data=JSON.parse(readFileSync(path,'utf8')) as Obj[];
 return data.map(s=>({sample_id:String(s.sample_id),group_id:String(s.sample_id),benchmark:s.benchmark==='locomo'?'locomo':'memops',sessions:s.add_phase.sessions.map((x:Obj,i:number)=>({session_id:x.session_id??`session-${i}`,messages:x.messages.map((m:Obj)=>({role:String(m.role),content:String(m.content),timestamp:String(m.timestamp)}))})),questions:s.search_items.map((q:Obj)=>({qid:q.qid,question:q.question,gold_answer:q.gold_answer,gold_rubric:q.gold_rubric,options:q.options,category:q.category??'unknown'}))}));
}
export function chunks(messages:Message[],maxMessages=20,maxWords=2000):Message[][]{
  const out:Message[][]=[];let current:Message[]=[];let words=0;
  for(const m of messages){const count=m.content.trim().split(/\s+/).length;if(current.length&&(current.length>=maxMessages||words+count>maxWords)){out.push(current);current=[];words=0;}current.push(m);words+=count;}
  if(current.length)out.push(current);return out;
}
export function saveDataset(samples:Sample[],path:string):void{mkdirSync(dirname(path),{recursive:true});writeFileSync(path,JSON.stringify(samples,null,2)+'\n');}
