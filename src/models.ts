export async function completion(base:string,key:string,model:string,messages:{role:string;content:string}[],json=false):Promise<string>{
  let last:unknown;
  for(let attempt=0;attempt<3;attempt++){
    try{
      const r=await fetch(`${base.replace(/\/$/,'')}/chat/completions`,{method:'POST',headers:{Authorization:`Bearer ${key}`,'Content-Type':'application/json'},body:JSON.stringify({model,messages,max_completion_tokens:1800,...(json?{response_format:{type:'json_object'}}:{})}),signal:AbortSignal.timeout(90000)});
      if(!r.ok){if(r.status===429||r.status>=500){last=new Error(`Model HTTP ${r.status}`);await new Promise(r=>setTimeout(r,1000*(attempt+1)));continue;}throw new Error(`Model HTTP ${r.status}`);}
      const d=await r.json() as {choices?:{message?:{content?:string};finish_reason?:string}[]};const choice=d.choices?.[0];if(!choice?.message?.content||choice.finish_reason==='length')throw new Error('Incomplete model response');return choice.message.content;
    }catch(e){last=e;if(attempt===2)throw e;}
  }
  throw last;
}
