"""LoCoMo speaker-label sensitivity dataset; do not change roles or gold."""
import argparse,json,pathlib,re,hashlib
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
source=pathlib.Path(a.input);dest=pathlib.Path(a.output);samples=json.loads(source.read_text());changed=0
for sample in samples:
 if sample['benchmark']!='locomo':raise SystemExit('This sensitivity conversion is only for LoCoMo')
 for session in sample['sessions']:
  for message in session['messages']:
   message['content'],n=re.subn(r'^(\[Session time:[^\]]+\]\n)?[^\n:]{1,80}: ',lambda m:m[1] or '',message['content'],count=1);changed+=n
dest.write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n');dest.with_suffix('.manifest.json').write_text(json.dumps({'conversion':'remove speaker-name prefix only; keep roles, timestamps, captions, original source IDs, questions and gold','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'changed_messages':changed},indent=2)+'\n');print(json.dumps({'changed_messages':changed}))
