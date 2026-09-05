"""Evaluation transport bridge: remote SSE -> one OpenAI JSON response.

Keeps the original model/messages/options. Lets fixed upstream clients whose
non-streaming connections are dropped use the same remote transport as candidate.
No credentials or message bodies are logged. Bind loopback only.
"""
import json, os, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_POST(self):
        try:
            if self.path != '/v1/chat/completions':
                self.send_error(404); return
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            body['stream']=True
            body['stream_options']={'include_usage':True}
            request=urllib.request.Request(os.environ['MEMORY_LLM_BASE_URL'].rstrip('/')+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':self.headers.get('Authorization','')})
            content=''; finish=None; usage=None; model=body['model']
            with urllib.request.urlopen(request, timeout=110) as response:
                for line in response:
                    if not line.startswith(b'data:'): continue
                    value=line[5:].strip()
                    if not value or value==b'[DONE]': continue
                    event=json.loads(value); choice=(event.get('choices') or [{}])[0]
                    content+=choice.get('delta',{}).get('content') or ''
                    finish=choice.get('finish_reason') or finish
                    usage=event.get('usage') or usage
            if finish!='stop': raise ValueError('Incomplete upstream stream')
            result={'id':'transport-bridge','object':'chat.completion','model':model,'choices':[{'index':0,'message':{'role':'assistant','content':content},'finish_reason':finish}],'usage':usage}
            payload=json.dumps(result).encode(); self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        except Exception as error:
            # Never interpolate a request/header/body in diagnostics.
            payload=json.dumps({'error':{'type':type(error).__name__,'message':'Upstream streaming transport failed'}}).encode()
            self.send_response(502);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(payload)

if __name__=='__main__':
    server=ThreadingHTTPServer(('127.0.0.1',int(os.environ.get('BRIDGE_PORT','8767'))),Handler)
    print('Streaming transport bridge ready',flush=True);server.serve_forever()
