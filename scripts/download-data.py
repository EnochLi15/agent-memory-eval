"""Explicit, pinned public dataset acquisition; never runs inside the service."""
import pathlib,json,urllib.request,subprocess,hashlib
root=pathlib.Path(__file__).resolve().parents[1];data=root/'.data';data.mkdir(exist_ok=True)
manifest=json.loads((root/'configs'/'datasets.json').read_text());expected={x['path']:x['sha256'] for x in manifest['files']}
locomo=data/'locomo';locomo.mkdir(exist_ok=True)
for name in ['conversations.jsonl','questions.jsonl']:
 p=locomo/name;key='locomo/'+name
 if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected[key]:
  with urllib.request.urlopen(f"https://raw.githubusercontent.com/mem-eval-suite/LoCoMo_refined/{manifest['locomo_revision']}/data/public/{name}",timeout=90) as r:p.write_bytes(r.read())
repo=data/'MemOps'
if not repo.exists():subprocess.run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/MemTensor/MemOps',str(repo)],check=True)
subprocess.run(['git','sparse-checkout','init','--cone'],cwd=repo,check=True)
subprocess.run(['git','sparse-checkout','set','generated_result/4-inject_evidence_with_distractors'],cwd=repo,check=True)
head=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,text=True,capture_output=True).stdout.strip()
if head!=manifest['memops_revision']:
 if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():raise SystemExit('Cached upstream has local changes; refusing checkout')
 subprocess.run(['git','fetch','origin',manifest['memops_revision']],cwd=repo,check=True)
 subprocess.run(['git','checkout','--detach',manifest['memops_revision']],cwd=repo,check=True)
for relative,digest in expected.items():
 p=data/relative
 if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise SystemExit('Dataset checksum mismatch: '+relative)
print('All',len(expected),'public source files match pinned checksums.')
subprocess.run(['node','dist/cli.js','prepare','--benchmark','locomo','--conversations',str(locomo/'conversations.jsonl'),'--questions',str(locomo/'questions.jsonl'),'--output',str(data/'locomo-normalized.json')],cwd=root,check=True)
subprocess.run(['node','dist/cli.js','prepare','--benchmark','memops','--directory',str(repo/'generated_result/4-inject_evidence_with_distractors'),'--output',str(data/'memops-normalized.json')],cwd=root,check=True)
subprocess.run(['python3','scripts/select-data.py'],cwd=root,check=True)
