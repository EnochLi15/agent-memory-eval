"""Deterministic group-disjoint dev/test selection; no model-dependent filtering."""
import collections
import hashlib
import json
import pathlib
import random

root = pathlib.Path(__file__).resolve().parents[1]
seed = 20260905
manifest = {'version':'group-disjoint-v1','seed':seed,'policy':'20% groups dev; test questions stratified round-robin by category and sample; entire dialogue retained','datasets':{}}
for benchmark in ['locomo','memops']:
    path=root/'.data'/f'{benchmark}-normalized.json'
    samples=json.loads(path.read_text())
    groups=sorted(set(s['group_id'] for s in samples),key=lambda x:hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())
    devgroups=set(groups[:max(1,len(groups)//5)])
    selected={}
    for partition,limit in [('dev',30),('test',500)]:
        pool=[s for s in samples if (s['group_id'] in devgroups)==(partition=='dev')]
        rng=random.Random(seed);rng.shuffle(pool)
        queues=collections.defaultdict(list)
        for sample in pool:
            qs=list(sample['questions']);rng.shuffle(qs)
            for q in qs:queues[q['category']].append((sample,q))
        kept=collections.defaultdict(list);count=0
        while count<limit and any(queues.values()):
            for category in sorted(queues):
                if not queues[category] or count>=limit:continue
                sample,q=queues[category].pop(0);kept[sample['sample_id']].append(q);count+=1
        result=[{**s,'questions':kept[s['sample_id']]} for s in pool if s['sample_id'] in kept]
        output=root/'.data'/f'{benchmark}-{partition}.json';output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        selected[partition]={'questions':count,'samples':len(result),'groups':sorted(set(s['group_id'] for s in result)),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'qids':[q['qid'] for s in result for q in s['questions']]}
    manifest['datasets'][benchmark]={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),**selected}
(root/'configs'/'splits.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({b:{p:{k:v for k,v in d[p].items() if k in ('questions','samples')} for p in ('dev','test')} for b,d in manifest['datasets'].items()}))
