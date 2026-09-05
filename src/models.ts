export async function completion(base:string,key:string,model:string,messages:{role:string;content:string}[],json=false):Promise<string>{
  let last:unknown;
  for(let attempt=0;attempt<3;attempt++){
    try{
      const r=await fetch(`${base.replace(/\/$/,'')}/chat/completions`,{method:'POST',headers:{Authorization:`Bearer ${key}`,'Content-Type':'application/json'},body:JSON.stringify({model,messages,max_completion_tokens:1800,stream:true,...(json?{response_format:{type:'json_object'}}:{})}),signal:AbortSignal.timeout(90000)});
      if(!r.ok){if(r.status===429||r.status>=500){last=new Error(`Model HTTP ${r.status}`);await new Promise(r=>setTimeout(r,1000*(attempt+1)));continue;}throw new Error(`Model HTTP ${r.status}`);}
      if(r.headers.get('content-type')?.includes('text/event-stream')){
        if(!r.body)throw new Error('Empty model stream');
        const decoder=new TextDecoder();let buffer='',content='',finish:string|null=null;
        for await(const chunk of r.body){buffer+=decoder.decode(chunk,{stream:true});buffer=buffer.replace(/\r\n/g,'\n');let end:number;while((end=buffer.indexOf('\n\n'))>=0){const event=buffer.slice(0,end);buffer=buffer.slice(end+2);for(const line of event.split('\n')){if(!line.startsWith('data:'))continue;const value=line.slice(5).trim();if(!value||value==='[DONE]')continue;const d=JSON.parse(value) as {choices?:{delta?:{content?:string};finish_reason?:string|null}[]};content+=d.choices?.[0]?.delta?.content??'';finish=d.choices?.[0]?.finish_reason??finish;}}}
        if(!content||finish!=='stop')throw new Error('Incomplete model stream');return content;
      }
      const d=await r.json() as {choices?:{message?:{content?:string};finish_reason?:string}[]};const choice=d.choices?.[0];if(!choice?.message?.content||choice.finish_reason==='length')throw new Error('Incomplete model response');return choice.message.content;
    }catch(e){last=e;if(attempt===2)throw e;}
  }
  throw last;
}
