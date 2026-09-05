import {readFileSync} from 'node:fs';
import {Ajv} from 'ajv';
import addFormats from 'ajv-formats';
import type {AddRequest,SearchRequest,Evidence} from './types.js';

const contract=JSON.parse(readFileSync(new URL('../contracts/contract.json',import.meta.url),'utf8')) as {schemas:Record<string,object>};
const ajv=new Ajv({allErrors:true,strict:false});(addFormats as unknown as (a:Ajv)=>void)(ajv);
const validators=Object.fromEntries(Object.entries(contract.schemas).map(([k,v])=>[k,ajv.compile(v)]));
export class MemoryClient{
  constructor(readonly baseUrl:string,private timeout=120000){}
  async health():Promise<void>{const r=await fetch(`${this.baseUrl}/health`,{signal:AbortSignal.timeout(this.timeout)});if(!r.ok)throw new Error(`Health HTTP ${r.status}`);}
  private async post(path:string,body:unknown,reqSchema:string,resSchema:string):Promise<unknown>{
    if(!validators[reqSchema]!(body))throw new Error(`Invalid outgoing ${reqSchema}: ${ajv.errorsText(validators[reqSchema]!.errors)}`);
    const r=await fetch(`${this.baseUrl}${path}`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:AbortSignal.timeout(this.timeout)});
    const data:unknown=await r.json();if(!r.ok)throw new Error(`${path}: HTTP ${r.status} ${JSON.stringify(data).slice(0,500)}`);
    if(!validators[resSchema]!(data))throw new Error(`Invalid incoming ${resSchema}`);return data;
  }
  async add(body:AddRequest):Promise<void>{
    const receipt=await this.post('/add',body,'addRequest','addResponse') as Record<string,unknown>;
    for(const k of ['request_id','user_id','session_id'] as const)if(receipt[k]!==body[k])throw new Error(`Receipt ${k} does not echo request`);
  }
  async search(body:SearchRequest):Promise<Evidence[]>{
    const response=await this.post('/search',body,'searchRequest','searchResponse') as {data:Evidence[]};
    if(response.data.length>Math.min(100,Math.floor(body.top_k)))throw new Error('Search exceeded top_k');
    if(response.data.some((e,i)=>i>0&&response.data[i-1]!.score<e.score))throw new Error('Search is not relevance-sorted');return response.data;
  }
}
