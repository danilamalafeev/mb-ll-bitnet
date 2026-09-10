#!/usr/bin/env python3
"""Frozen E28 length transfer. Inference only; every output directory is exclusive."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import torch
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from looped_bitnet import w4_e27 as w4, continuation_e24 as fl, longer_native8_e20 as old
from looped_bitnet.register_e15 import evaluate_program
from scripts.composition_e21 import _strata

ROOT = w4.ROOT
STATES = [[x, y] for x in range(16) for y in range(16)]
KEYS = w4.COST_KEYS
BUDGET = dict(zip(KEYS, (72, 18432, 82944, 663552)))
SOURCES = ['scripts/length_transfer_e28.py', 'tests/test_length_transfer_e28.py']

def read(path):
    return json.loads(Path(path).read_text())

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def trace(program, state):
    x, y = state
    result = []
    for op in program:
        if op == 'ADD': x = (x + y) % 16
        elif op == 'XOR': x = x ^ y
        elif op == 'SWAP': x, y = y, x
        else: raise ValueError('opcode')
        result.append([x, y])
    return result

def selection(root=ROOT):
    frozen = read(root / 'results/E28_SELECTION.json')
    if frozen['schema'] != 'e28_root_selection_v1' or frozen['operators'] != ['ADD','XOR','SWAP'] or frozen['state_order'] != STATES:
        raise ValueError('selection schema/states/operators')
    funcs = lambda p: [trace(p, s)[-1] if p else s for s in STATES]
    shorter = set()
    expected = []; audit = {}
    for length in range(6):
        candidates = [(p, hashlib.sha256(bytes(v for s in funcs(p) for v in s)).hexdigest()) for p in itertools.product(frozen['operators'], repeat=length)]
        if length in (4,5):
            eligible = [(p, d) for p, d in candidates if d not in shorter]
            audit[str(length)] = dict(eligible_syntax_count=len(eligible), eligible_semantic_classes=len({d for p,d in eligible}), shorter_function_classes=len(shorter))
            ranked = sorted(eligible, key=lambda v: (hashlib.sha256(('E28-v1:'+','.join(v[0])).encode()).hexdigest(), v[0]))
            seen = set(); picked = []
            for p,d in ranked:
                if d not in seen: picked.append((p,d)); seen.add(d)
            for i,(p,d) in enumerate(picked[:6],1):
                expected.append(dict(name=f'L{length}_{i}', length=length, program=list(p), rank_sha256=hashlib.sha256(('E28-v1:'+','.join(p)).encode()).hexdigest(), function_sha256=d, cases=[dict(state=s, target_trace=trace(p,s)) for s in STATES]))
        shorter.update(d for p,d in candidates)
    if frozen['programs'] != expected or frozen['audit'] != audit:
        raise ValueError('selection/targets/novelty changed')
    return frozen

def verify(root=ROOT):
    parent = read(root / 'results/E27_PROTECTED_HASHES.json')['sha256']
    reference = read(root / 'results/E28_REFERENCE.json')['sha256']
    guard = read(root / 'results/E28_PROTECTED_HASHES.json')['sha256']
    required = {**parent, **reference, 'results/E28_REFERENCE.json': old.sha256_file(root/'results/E28_REFERENCE.json')}
    for path in ('scripts/w4_e27.py', 'looped_bitnet/w4_e27.py', 'tests/test_w4_e27.py'):
        if path not in guard: raise ValueError('guard missing E27 source '+path)
    for path, sha in required.items():
        if guard.get(path) != sha: raise ValueError('guard missing/mismatched '+path)
    for path, sha in guard.items():
        if old.sha256_file(root/path) != sha: raise ValueError('protected changed '+path)
    return guard

def manifest(root=ROOT):
    return dict(schema='e28_length_v1', selection=selection(root), protected=verify(root), source_hashes={p:old.sha256_file(root/p) for p in SOURCES}, budget=BUDGET, seeds=[0,1,2], models=['E27_W4','E24_float'], native_steps=8)

def write(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2)

def preflight(out=Path('runs/e28_length_preflight'), root=ROOT):
    m = manifest(root)
    (root/out).mkdir(parents=True, exist_ok=False)
    write(root/out/'manifest.json', m)
    return m

def metrics(items):
    length = len(items[0]['target_trace'])
    correct = [[a==b for a,b in zip(p['predicted_trace'], p['target_trace'])] for p in items]
    hist = {str(i):0 for i in range(1,length+1)}; hist['none']=0
    for flags in correct: hist[next((str(i+1) for i,c in enumerate(flags) if not c),'none')] += 1
    return dict(denominator=len(items), final_joint=sum(c[-1] for c in correct), full_trace=sum(all(c) for c in correct), prefix_joint=[sum(c[i] for c in correct) for i in range(length)], final_x=sum(p['predicted_trace'][-1][0]==p['target_trace'][-1][0] for p in items), final_y=sum(p['predicted_trace'][-1][1]==p['target_trace'][-1][1] for p in items), first_divergence=hist, recovery=sum(c[-1] and not all(c) for c in correct))

def score(rows, selected, strata):
    if [r['program'] for r in rows] != [p['program'] for p in selected]: raise ValueError('program scope/order')
    membership = {tuple(s):k for k,ss in strata.items() for s in ss}
    if sum(len(ss) for ss in strata.values()) != len(membership): raise ValueError('overlapping strata')
    if set(membership) != set(map(tuple,STATES)) or {k:len(v) for k,v in strata.items()} != dict(train=192,validation=32,test=32): raise ValueError('strata')
    result = []
    for row,p in zip(rows,selected):
        preds=row['predictions']
        if [v['state'] for v in preds] != STATES: raise ValueError('state order')
        for v in preds:
            target=trace(p['program'],v['state']); pred=v['predicted_trace']
            if len(pred)!=len(target) or any(not isinstance(t,list) or len(t)!=2 or any(type(a) is not int or not 0<=a<16 for a in t) for t in pred): raise ValueError('malformed prediction')
            flags=[a==b for a,b in zip(pred,target)]
            if any(type(b) is not bool for b in v['prefix_joint_correct']): raise ValueError('correctness flag types')
            if v['target_trace'] != target or v['prefix_joint_correct'] != flags or type(v['joint_final_correct']) is not bool or v['joint_final_correct'] != flags[-1]: raise ValueError('target/correctness')
            if v.get('stratum') != membership[tuple(v['state'])]: raise ValueError('stratum identity')
        slices={'all':preds, **{k:[v for v in preds if v['stratum']==k] for k in strata}}
        ms={k:metrics(v) for k,v in slices.items()}
        result.append(dict(name=p['name'],program=p['program'],length=p['length'],metrics=ms,predictions=preds))
    return dict(rows=result, primary_by_length={str(l):all(r['metrics']['all']['final_joint']>=244 for r in result if r['length']==l) for l in (4,5)}, secondary_by_length={str(l):all(r['metrics']['all']['full_trace']>=244 for r in result if r['length']==l) for l in (4,5)})

def paired(a,b):
    if [(r['name'],r['program']) for r in a['rows']] != [(r['name'],r['program']) for r in b['rows']]: raise ValueError('paired programs')
    result=[]
    for ar,br in zip(a['rows'],b['rows']):
        if [(v['state'],v['target_trace'],v['stratum']) for v in ar['predictions']] != [(v['state'],v['target_trace'],v['stratum']) for v in br['predictions']]: raise ValueError('paired identity')
        slices={}
        for stratum in ('all','train','validation','test'):
            metrics_pair={}
            for metric in ('final_joint','full_trace'):
                counts=dict(W4_wins=0,float_wins=0,both_correct=0,both_wrong=0)
                for x,y in zip(ar['predictions'],br['predictions']):
                    if stratum!='all' and x['stratum']!=stratum: continue
                    flags=lambda v: v['joint_final_correct'] if metric=='final_joint' else all(v['prefix_joint_correct'])
                    ac,bc=flags(x),flags(y)
                    counts['both_correct' if ac and bc else 'W4_wins' if ac else 'float_wins' if bc else 'both_wrong']+=1
                metrics_pair[metric]=counts
            slices[stratum]=metrics_pair
        result.append(dict(name=ar['name'],metrics=slices))
    return result

def load(label, seed, root=ROOT):
    directory='runs/e27_w4' if label=='E27_W4' else 'runs/e24_continuation'
    if label not in ('E27_W4','E24_float') or seed not in (0,1,2): raise ValueError('model scope')
    with torch.random.fork_rng(devices=[]):
        loader=w4.load_checkpoint if label=='E27_W4' else fl.load_checkpoint
        return loader(root/directory/f'seed{seed}/u16000.pt',read(root/directory/'manifest.json'),seed=seed)

def evaluate(model, optimizer, programs, states, record):
    before=(old.digest_state_dict(model),old.digest_object(optimizer.state_dict()),torch.get_rng_state().clone(),model.training)
    def count(key,args):
        n,l=int(args[0].shape[0]),int(args[2].shape[1])
        for k,v in zip(KEYS,(1,n,n*l,n*l*8)): record[key][k]+=v
    def post(m,args,out):
        count('completed',args)
        if not isinstance(out,tuple) or len(out)!=2 or any(not torch.isfinite(t).all() for t in out): raise FloatingPointError('nonfinite output')
    pre=model.register_forward_pre_hook(lambda m,args:count('attempted',args)); hook=model.register_forward_hook(post)
    rows=[]; model.eval()
    try:
        with torch.inference_mode(),torch.autocast('cpu',enabled=False):
            for program in programs:
                rows.append(evaluate_program(model,program,states,include_predictions=True))
                record['partial_rows']=rows
    finally:
        pre.remove(); hook.remove(); model.train(before[3])
        if old.digest_state_dict(model)!=before[0] or old.digest_object(optimizer.state_dict())!=before[1] or not torch.equal(torch.get_rng_state(),before[2]) or model.training!=before[3]: raise ValueError('evaluation mutated model/optimizer/RNG/mode')
    record['unchanged']=True
    return rows

def execute(out, preflight_path=Path('runs/e28_length_preflight'), root=ROOT, qa=False):
    out=root/Path(out)
    if out.exists(): raise FileExistsError(out)
    m=manifest(root)
    if not qa and read(root/preflight_path/'manifest.json')!=m: raise ValueError('frozen preflight changed')
    out.mkdir(parents=True,exist_ok=False); write(out/'manifest.json',m)
    report=dict(status='running',qa=qa,training_updates=0,models={})
    try:
        strata=_strata(root); membership={tuple(s):k for k,ss in strata.items() for s in ss}
        for label in m['models']:
            report['models'][label]={}
            for seed in m['seeds']:
                rec=dict(attempted=dict.fromkeys(KEYS,0),completed=dict.fromkeys(KEYS,0)); report['models'][label][str(seed)]=rec
                model,opt,_=load(label,seed,root)
                programs=[('ADD',)] if qa else [p['program'] for p in m['selection']['programs']]
                states=list(map(tuple,STATES[:2] if qa else STATES))
                rows=evaluate(model,opt,programs,states,rec)
                if not qa:
                    for row in rows:
                        for v in row['predictions']: v['stratum']=membership[tuple(v['state'])]
                    rec['evaluation']=score(rows,m['selection']['programs'],strata)
                write(out/f'{label}_seed{seed}.json',rec)
        if not qa:
            report['paired']={str(s):paired(report['models']['E27_W4'][str(s)]['evaluation'],report['models']['E24_float'][str(s)]['evaluation']) for s in m['seeds']}
            report['primary_conjunction']={label:all(all(r['evaluation']['primary_by_length'].values()) for r in seeds.values()) for label,seeds in report['models'].items()}
            report['secondary_conjunction']={label:all(all(r['evaluation']['secondary_by_length'].values()) for r in seeds.values()) for label,seeds in report['models'].items()}
        verify(root); report['status']='complete'
    except Exception as exc:
        report['status']='suspended'; report['error']=f'{type(exc).__name__}: {exc}'
    finally:
        report['cost']={phase:{k:sum(r[phase][k] for seeds in report['models'].values() for r in seeds.values()) for k in KEYS} for phase in ('attempted','completed')}
        if report['status']=='complete' and not qa and (report['cost']['completed']!=BUDGET or report['cost']['attempted']!=BUDGET): report.update(status='suspended',error='budget mismatch')
        write(out/'report.json',report)
    return report

def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['preflight','evaluate','qa']); p.add_argument('--out',type=Path,required=True); p.add_argument('--preflight',type=Path,default=Path('runs/e28_length_preflight')); a=p.parse_args()
    result=preflight(a.out) if a.mode=='preflight' else execute(a.out,a.preflight,qa=a.mode=='qa')
    print(json.dumps({'status':result.get('status','preflight_ready'),'cost':result.get('cost')}))
    if result.get('status')=='suspended': raise SystemExit(1)
if __name__=='__main__': main()
