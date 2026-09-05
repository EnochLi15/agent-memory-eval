export type Message={role:string;content:string;timestamp:string};
export type AddRequest={request_id:string;user_id:string;session_id:string;messages:Message[]};
export type SearchRequest={query:string;user_id:string;top_k:number;options?:unknown[]};
export type Evidence={id:string;content:string;score:number;created_at:string};
export type Question={qid:string;question:string;gold_answer:string|string[];gold_rubric?:Record<string,unknown>;options?:unknown[];category:string;gold_evidence?:string[]};
export type Sample={sample_id:string;group_id:string;benchmark:'locomo'|'memops'|'synthetic';sessions:{session_id:string;messages:Message[]}[];questions:Question[]};
export type EvalConfig={baseUrl:string;runDir:string;runId:string;dataFile:string;limit:number;concurrency:number;mode:'proxy'|'upstream-reproduction'|'competition-reproduction';answerModel:string;judgeModel:string;llmBase:string;llmKey:string;judgeBase:string;judgeKey:string;judgeKind:'rubric'|'refined-python'|'proxy';maxMessages:number;maxWords:number;topK:number;resume:boolean};
