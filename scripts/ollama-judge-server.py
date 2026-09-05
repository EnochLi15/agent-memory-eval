"""Evaluation-only OpenAI adapter: preserves judge messages; maps thinking=false to Ollama native API."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json, os, time, urllib.request
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != '/v1/chat/completions': self.send_error(404); return
        request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        payload={'model':request['model'],'messages':request['messages'],'stream':False,'think':False,'keep_alive':'10m','options':{'temperature':request.get('temperature',0),'num_ctx':8192,'num_predict':512}}
        try:
            req=urllib.request.Request(os.getenv('OLLAMA_URL','http://127.0.0.1:11434')+'/api/chat',json.dumps(payload).encode(),{'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=170) as r: result=json.load(r)
            data={'id':'local-judge','object':'chat.completion','created':int(time.time()),'model':request['model'],'choices':[{'index':0,'message':{'role':'assistant','content':result['message']['content']},'finish_reason':'stop'}],'usage':{'prompt_tokens':result.get('prompt_eval_count',0),'completion_tokens':result.get('eval_count',0),'total_tokens':result.get('prompt_eval_count',0)+result.get('eval_count',0)}}
            body=json.dumps(data).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        except Exception as error:
            self.send_error(503,type(error).__name__)
    def log_message(self, *args): pass
if __name__ == '__main__':
    print('Evaluation judge adapter ready at 127.0.0.1:8766',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8766),Handler).serve_forever()
