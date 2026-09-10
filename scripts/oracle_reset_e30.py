#!/usr/bin/env python3
"""E30 frozen fresh-context oracle diagnostic; no training and exclusive artifacts."""
import argparse
import json
import random
from pathlib import Path
import sys
import torch
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import length_transfer_e28 as e

ROOT=e.ROOT
MODELS=['E27_W4','E24_float']; SEEDS=[0,1,2]; OPS=['XOR','SWAP']
BUDGET=dict(zip(e.KEYS,(12,3072,3072,24576)))
SOURCES=['scripts/oracle_reset_e30.py','tests/test_oracle_reset_e30.py']

def verify(root=ROOT):
    prior=e.verify(root)
    ref=e.read(root/'results/E30_REFERENCE.json')['sha256']
    required={**prior,**ref,'results/E30_REFERENCE.json':e.old.sha256_file(root/'results/E30_REFERENCE.json')}
    guard=e.read(root/'results/E30_PROTECTED_HASHES.json')['sha256']
    for path,sha in required.items():
        if guard.get(path)!=sha: raise ValueError('guard missing/mismatch '+path)
    for path,sha in guard.items():
        if e.old.sha256_file(root/path)!=sha: raise ValueError('protected changed '+path)
    return guard

def saved(root=ROOT):
    selected=e.selection(root)['programs']; strata=e._strata(root)
    report=e.read(root/'runs/e28_length/report.json')
    if report['status']!='complete' or report['qa'] is not False or report['training_updates']!=0 or set(report['models'])!=set(MODELS): raise ValueError('saved report identity')
    cells={}
    for label in MODELS:
        if set(report['models'][label])!={'0','1','2'}: raise ValueError('seed scope')
        for seed in SEEDS:
            rec=report['models'][label][str(seed)]
            result=e.score(rec['evaluation']['rows'],selected,strata)
            if result!=rec['evaluation']: raise ValueError('saved metrics identity')
            rows=[r for r in result['rows'] if r['length']==5]
            for row in rows:
                if row['program'][-1] not in OPS: raise ValueError('last opcode')
                if len({tuple(e.trace(row['program'][:4],s)[-1]) for s in e.STATES})!=256: raise ValueError('prefix not bijective')
            if len(rows)!=6: raise ValueError('L5 scope')
            cells[label,seed]=rows
    return cells

def manifest(root=ROOT):
    guard=verify(root); saved(root)
    return dict(schema='e30_oracle_v1',protected=guard,source_hashes={p:e.old.sha256_file(root/p) for p in SOURCES},models=MODELS,seeds=SEEDS,opcodes=OPS,states=e.STATES,native_steps=8,budget=BUDGET,lifted_cases=9216,training_updates=0)

def preflight(out,root=ROOT):
    out=root/Path(out)
    if out.exists(): raise FileExistsError(out)
    m=manifest(root); out.mkdir(parents=True,exist_ok=False);e.write(out/'manifest.json',m);return m

def evaluate(model,opt,programs,states,record):
    modes=[(m,m.training) for m in model.modules()]
    py=random.getstate()
    try:
        return e.evaluate(model,opt,programs,states,record)
    finally:
        # E28 restores the root mode recursively; restore heterogeneous child modes too.
        for m,mode in modes: m.training=mode
        unchanged=random.getstate()==py
        record['all_modes_restored']=all(m.training==mode for m,mode in modes)
        record['external_rng_unchanged']=unchanged
        if not unchanged: raise ValueError('evaluation mutated external RNG')

def primitive_table(rows):
    if len(rows)!=2 or {tuple(r['program']) for r in rows}!={('XOR',),('SWAP',)}: raise ValueError('fresh opcode scope')
    table={}
    for row in rows:
        op=row['program'][0]
        for p in row['predictions']:
            state=p['state']
            if not isinstance(state,list) or len(state)!=2 or any(type(v)is not int or not 0<=v<16 for v in state): raise ValueError('fresh state')
            key=(op,*state)
            if key in table: raise ValueError('duplicate fresh key')
            target=e.trace([op],state); pred=p['predicted_trace']
            if len(pred)!=1 or not isinstance(pred[0],list) or len(pred[0])!=2 or any(type(v)is not int or not 0<=v<16 for v in pred[0]): raise ValueError('fresh prediction')
            flag=pred==target
            if p['target_trace']!=target or any(type(v)is not bool for v in p['prefix_joint_correct']) or p['prefix_joint_correct']!=[flag] or type(p['joint_final_correct'])is not bool or p['joint_final_correct']!=flag: raise ValueError('fresh target/flags')
            table[key]=p
    if set(table)!={(op,*s) for op in OPS for s in e.STATES}: raise ValueError('fresh state scope')
    return table

def pairs(cases):
    counts=dict(recovered=0,introduced=0,both_correct=0,both_wrong=0)
    for c in cases:
        a,b=c['long_final_correct'],c['fresh_correct']
        counts['both_correct' if a and b else 'introduced' if a else 'recovered' if b else 'both_wrong']+=1
    errors=counts['recovered']+counts['both_wrong']
    return dict(denominator=len(cases),**counts,long_correct=counts['both_correct']+counts['introduced'],fresh_correct=counts['both_correct']+counts['recovered'],saved_errors=errors,rescued_fraction=counts['recovered']/errors if errors else None)

def summaries(cases):
    return {subset:{stratum:pairs([c for c in cases if (subset=='all' or c['prefix4_all_correct']==(subset=='prefix4_correct')) and (stratum=='all' or c['initial_stratum']==stratum)]) for stratum in ('all','train','validation','test')} for subset in ('all','prefix4_correct','prefix4_wrong')}

def lift(rows,table):
    cases=[]
    for row in rows:
        for p in row['predictions']:
            target=e.trace(row['program'],p['state']); true=target[3]; op=row['program'][-1]
            fresh=table[(op,*true)]
            if fresh['target_trace'][0]!=target[-1]: raise ValueError('oracle target mismatch')
            cases.append(dict(program_name=row['name'],program=row['program'],initial_state=p['state'],initial_stratum=p['stratum'],true_prefix_state=true,last_opcode=op,target_trace=target,saved_long_trace=p['predicted_trace'],long_final_correct=p['joint_final_correct'],prefix4_all_correct=all(p['prefix_joint_correct'][:4]),fresh_input_state=fresh['state'],fresh_prediction=fresh['predicted_trace'][0],fresh_correct=fresh['joint_final_correct']))
    return dict(cases=cases,paired=summaries(cases),programs={r['name']:dict(fresh_correct=sum(c['fresh_correct'] for c in cases if c['program_name']==r['name']),denominator=256,descriptive_244=sum(c['fresh_correct'] for c in cases if c['program_name']==r['name'])>=244) for r in rows})

def execute(out,preflight_path=Path('runs/e30_oracle_preflight'),root=ROOT,qa=False):
    out=root/Path(out)
    if out.exists(): raise FileExistsError(out)
    m=manifest(root)
    if not qa and e.read(root/preflight_path/'manifest.json')!=m: raise ValueError('preflight changed')
    old=saved(root);out.mkdir(parents=True,exist_ok=False);e.write(out/'manifest.json',m)
    report=dict(status='running',qa=qa,training_updates=0,models={})
    try:
        allcases=[]
        for label in MODELS:
            report['models'][label]={}
            for seed in SEEDS:
                rec={phase:dict.fromkeys(e.KEYS,0) for phase in ('attempted','completed')};report['models'][label][str(seed)]=rec
                model,opt,_=e.load(label,seed,root)
                rows=evaluate(model,opt,[['ADD']] if qa else [[op] for op in OPS],list(map(tuple,e.STATES[:2] if qa else e.STATES)),rec)
                rec['fresh_rows']=rows
                if not qa:
                    table=primitive_table(rows);rec['fresh_counts']={op:sum(v['joint_final_correct'] for k,v in table.items() if k[0]==op) for op in OPS}
                    rec['lifted']=lift(old[label,seed],table);allcases.extend(rec['lifted']['cases'])
                e.write(out/f'{label}_seed{seed}.json',rec)
        if not qa:
            if len(allcases)!=9216: raise ValueError('lifted scope')
            report['paired']=summaries(allcases); risk=report['paired']['prefix4_correct']['all']
            if risk['saved_errors']==0: raise ValueError('unexpected vacuous aggregate')
            report['fresh_exact_all']=all(v==256 for seeds in report['models'].values() for r in seeds.values() for v in r['fresh_counts'].values())
            report['late_error_rescue_all']=risk['both_wrong']==0 and risk['introduced']==0
            report['lifted_comparisons']=len(allcases)
        report['status']='complete'
    except Exception as exc:
        report.update(status='suspended',error=f'{type(exc).__name__}: {exc}')
    finally:
        try: verify(root);report['protected_unchanged']=True
        except Exception as exc: report.update(status='suspended',protected_unchanged=False,error=f'{type(exc).__name__}: {exc}')
        report['cost']={phase:{k:sum(r[phase][k] for seeds in report['models'].values() for r in seeds.values()) for k in e.KEYS} for phase in ('attempted','completed')}
        expected=dict(zip(e.KEYS,(6,12,12,96))) if qa else BUDGET
        if report['status']=='complete' and any(v!=expected for v in report['cost'].values()): report.update(status='suspended',error='budget mismatch')
        e.write(out/'report.json',report)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','evaluate','qa']);p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight',type=Path,default=Path('runs/e30_oracle_preflight'));a=p.parse_args()
    r=preflight(a.out) if a.mode=='preflight' else execute(a.out,a.preflight,qa=a.mode=='qa')
    print(json.dumps({'status':r.get('status','preflight_ready'),'cost':r.get('cost')}))
    if r.get('status')=='suspended': raise SystemExit(1)
if __name__=='__main__': main()
