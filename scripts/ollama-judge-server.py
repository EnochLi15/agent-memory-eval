"""Evaluation-only OpenAI adapter: preserves judge messages; maps thinking=false to Ollama native API."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import argparse, json, os, time, urllib.request, signal
from pathlib import Path
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/health': self.send_error(404); return
        body=json.dumps({'status':'ok','pid':os.getpid(),'protocol':'ollama-judge-adapter-v1'}).encode()
        self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_POST(self):
        if self.path != '/v1/chat/completions': self.send_error(404); return
        request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        # Long memory ingestion may be idle for more than ten minutes. Keep the
        # judge resident until explicit teardown so the next verdict cannot stall
        # behind embedding eviction while loading the large model again.
        payload={'model':request['model'],'messages':request['messages'],'stream':False,'think':False,'keep_alive':-1,'options':{'temperature':request.get('temperature',0),'num_ctx':8192,'num_predict':512}}
        try:
            req=urllib.request.Request(os.getenv('OLLAMA_URL','http://127.0.0.1:11434')+'/api/chat',json.dumps(payload).encode(),{'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=170) as r: result=json.load(r)
            data={'id':'local-judge','object':'chat.completion','created':int(time.time()),'model':request['model'],'choices':[{'index':0,'message':{'role':'assistant','content':result['message']['content']},'finish_reason':'stop'}],'usage':{'prompt_tokens':result.get('prompt_eval_count',0),'completion_tokens':result.get('eval_count',0),'total_tokens':result.get('prompt_eval_count',0)+result.get('eval_count',0)}}
            body=json.dumps(data).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        except Exception as error:
            print(json.dumps({'event':'judge_request_failed','at':time.time(),'error_type':type(error).__name__}),flush=True)
            self.send_error(503,type(error).__name__)
    def log_message(self, *args): pass
if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8766);parser.add_argument('--ready-file',type=Path);args=parser.parse_args()
    def interrupted(signum,_frame):
        print(json.dumps({'event':'judge_signal','signal':signum,'at':time.time()}),flush=True)
        raise SystemExit(128+signum)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    with ThreadingHTTPServer(('127.0.0.1',args.port),Handler) as server:
        ready={'pid':os.getpid(),'base_url':f'http://127.0.0.1:{server.server_port}/v1','protocol':'ollama-judge-adapter-v1'}
        if args.ready_file:
            temporary=args.ready_file.with_suffix('.tmp');temporary.write_text(json.dumps(ready)+'\n');temporary.replace(args.ready_file)
        print(json.dumps({'event':'judge_ready',**ready}),flush=True)
        try:server.serve_forever()
        finally:print(json.dumps({'event':'judge_stopped','at':time.time()}),flush=True)
