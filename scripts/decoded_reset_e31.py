#!/usr/bin/env python3
"""E31 own decoded prefix, fresh last-op execution. Frozen, inference only."""
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import oracle_reset_e30 as o
from scripts import length_transfer_e28 as e
ROOT=e.ROOT
MODELS=o.MODELS; SEEDS=o.SEEDS
BUDGET=dict(zip(e.KEYS,(36,9216,9216,73728)))
SOURCES=['scripts/decoded_reset_e31.py','tests/test_decoded_reset_e31.py']

def verify(root=ROOT):
    prior=o.verify(root)
    ref=e.read(root/'results/E31_REFERENCE.json')['sha256']
    required={**prior,**ref,'results/E31_REFERENCE.json':e.old.sha256_file(root/'results/E31_REFERENCE.json')}
    guard=e.read(root/'results/E31_PROTECTED_HASHES.json')['sha256']
    for path,sha in required.items():
        if guard.get(path)!=sha: raise ValueError('guard missing/mismatch '+path)
    for path,sha in guard.items():
        if e.old.sha256_file(root/path)!=sha: raise ValueError('protected changed '+path)
    return guard

def controls(root=ROOT):
    report=e.read(root/'runs/e30_oracle/report.json')
    if report['status']!='complete' or report['qa'] is not False or report['training_updates']!=0 or set(report['models'])!=set(MODELS): raise ValueError('E30 report identity')
    result={}
    for label in MODELS:
        if set(report['models'][label])!={'0','1','2'}: raise ValueError('E30 seeds')
        for seed in SEEDS: result[label,seed]=o.primitive_table(report['models'][label][str(seed)]['fresh_rows'])
    return result

def manifest(root=ROOT):
    guard=verify(root); o.saved(root); controls(root)
    return dict(schema='e31_decoded_v1',protected=guard,source_hashes={p:e.old.sha256_file(root/p) for p in SOURCES},models=MODELS,seeds=SEEDS,budget=BUDGET,native_steps=8,training_updates=0,input='saved predicted_trace[3]',correctness_target='original L5 final target')

def preflight(out,root=ROOT):
    out=root/Path(out)
    if out.exists(): raise FileExistsError(out)
    m=manifest(root);out.mkdir(parents=True,exist_ok=False);e.write(out/'manifest.json',m);return m

def state(value):
    return isinstance(value,list) and len(value)==2 and all(type(v)is int and 0<=v<16 for v in value)

def decoded_inputs(row):
    if row['length']!=5 or len(row['program'])!=5 or row['program'][-1] not in o.OPS: raise ValueError('L5 program')
    if [p['state'] for p in row['predictions']]!=e.STATES: raise ValueError('original state order')
    inputs=[]
    for p in row['predictions']:
        target=e.trace(row['program'],p['state']);pred=p['predicted_trace']
        if len(pred)!=5 or not all(state(s) for s in pred): raise ValueError('saved prediction')
        flags=[a==b for a,b in zip(pred,target)]
        if p['target_trace']!=target or p['prefix_joint_correct']!=flags or any(type(f)is not bool for f in p['prefix_joint_correct']) or type(p['joint_final_correct'])is not bool or p['joint_final_correct']!=flags[-1] or p['stratum'] not in ('train','validation','test'): raise ValueError('saved target/flags/stratum')
        inputs.append(tuple(pred[3]))
    return inputs

def map_cases(row,fresh_rows,table):
    inputs=decoded_inputs(row);op=row['program'][-1]
    if len(fresh_rows)!=1 or fresh_rows[0]['program']!=[op]: raise ValueError('fresh program scope')
    fresh=fresh_rows[0]['predictions']
    if len(fresh)!=256 or any(not state(p['state']) for p in fresh) or [tuple(p['state']) for p in fresh]!=inputs: raise ValueError('fresh input sequence')
    cases=[]
    for original,p in zip(row['predictions'],fresh):
        local=e.trace([op],p['state']);pred=p['predicted_trace']
        if len(pred)!=1 or not state(pred[0]): raise ValueError('fresh prediction')
        flag=pred==local
        if p['target_trace']!=local or p['prefix_joint_correct']!=[flag] or any(type(f)is not bool for f in p['prefix_joint_correct']) or type(p['joint_final_correct'])is not bool or p['joint_final_correct']!=flag: raise ValueError('fresh target/flags')
        target=original['target_trace']; own=original['predicted_trace'][3]
        lookup=table[(op,*own)];oracle=table[(op,*target[3])]
        correct=pred[0]==target[-1];prefix=all(original['prefix_joint_correct'][:4])
        cases.append(dict(program_name=row['name'],program=row['program'],initial_state=original['state'],initial_stratum=original['stratum'],target_trace=target,saved_long_trace=original['predicted_trace'],decoded4=own,true4=target[3],fresh_input_state=p['state'],fresh_prediction=pred[0],fresh_local_target=local[0],fresh_local_correct=flag,fresh_correct=correct,long_final_correct=original['joint_final_correct'],fourth_readout_correct=own==target[3],fourth_correct=own==target[3],prefix4_all_correct=prefix,hybrid_full_trace_correct=prefix and correct,direct_lookup_match=pred==lookup['predicted_trace'],e30_lookup_prediction=lookup['predicted_trace'][0],e30_oracle_prediction=oracle['predicted_trace'][0],e30_oracle_correct=oracle['predicted_trace'][0]==target[-1]))
    return cases

def comparison(cases):
    result=o.pairs(cases)
    result['own_vs_oracle']=o.pairs([dict(long_final_correct=c['e30_oracle_correct'],fresh_correct=c['fresh_correct']) for c in cases])
    result['wrong_prefix_correct_fourth']=sum(not c['prefix4_all_correct'] and c['fourth_correct'] for c in cases)
    result['lost_old_recoveries']=sum(not c['prefix4_all_correct'] and c['long_final_correct'] and not c['fresh_correct'] for c in cases)
    return result

def summaries(cases):
    predicates={'all':lambda c:True,'fourth_correct':lambda c:c['fourth_correct'],'fourth_wrong':lambda c:not c['fourth_correct'],'prefix4_correct':lambda c:c['prefix4_all_correct'],'prefix4_wrong':lambda c:not c['prefix4_all_correct']}
    return {subset:{s:comparison([c for c in cases if predicate(c) and (s=='all' or c['initial_stratum']==s)]) for s in ('all','train','validation','test')} for subset,predicate in predicates.items()}

def program_summary(cases):
    if len(cases)!=256: raise ValueError('program denominator')
    final=sum(c['fresh_correct'] for c in cases);full=sum(c['hybrid_full_trace_correct'] for c in cases)
    return dict(denominator=256,fresh_correct=final,hybrid_full_trace_correct=full,descriptive_244=final>=244,hybrid_full_trace_244=full>=244,paired=summaries(cases))

def execute(out,preflight_path=Path('runs/e31_decoded_preflight'),root=ROOT,qa=False):
    out=root/Path(out)
    if out.exists(): raise FileExistsError(out)
    m=manifest(root)
    if not qa and e.read(root/preflight_path/'manifest.json')!=m: raise ValueError('preflight changed')
    saved=o.saved(root);tables=controls(root)
    out.mkdir(parents=True,exist_ok=False);e.write(out/'manifest.json',m)
    report=dict(status='running',qa=qa,training_updates=0,models={})
    try:
        allcases=[]
        for label in MODELS:
            report['models'][label]={}
            for seed in SEEDS:
                rec={phase:dict.fromkeys(e.KEYS,0) for phase in ('attempted','completed')};report['models'][label][str(seed)]=rec
                rec.update(fresh_rows=[],cases=[],programs={})
                model,opt,_=e.load(label,seed,root)
                if qa: rec['fresh_rows']=o.evaluate(model,opt,[['ADD']],list(map(tuple,e.STATES[:2])),rec)
                else:
                    for row in saved[label,seed]:
                        fresh=o.evaluate(model,opt,[[row['program'][-1]]],decoded_inputs(row),rec)
                        rec['fresh_rows'].append(dict(program_name=row['name'],row=fresh[0]))
                        cases=map_cases(row,fresh,tables[label,seed]);rec['cases'].extend(cases)
                        rec['programs'][row['name']]=program_summary(cases)
                    rec['paired']=summaries(rec['cases'])
                    risk=rec['paired']['prefix4_correct']['all']
                    rec['late_error_rescue_all']=risk['both_wrong']==0 and risk['introduced']==0
                    allcases.extend(rec['cases'])
                e.write(out/f'{label}_seed{seed}.json',rec)
        if not qa:
            if len(allcases)!=9216: raise ValueError('case scope')
            report['paired']=summaries(allcases);risk=report['paired']['prefix4_correct']['all']
            if risk['saved_errors']==0: raise ValueError('vacuous late-error diagnostic')
            report['late_error_rescue_all']=risk['both_wrong']==0 and risk['introduced']==0
            report['direct_lookup_match_all']=all(c['direct_lookup_match'] for c in allcases)
            report['fourth_equivalence_expected_control']=all(c['fresh_correct']==c['fourth_correct'] for c in allcases)
            report['descriptive_conjunction']={label:all(p['descriptive_244'] for rec in seeds.values() for p in rec['programs'].values()) for label,seeds in report['models'].items()}
            report['hybrid_full_trace_conjunction']={label:all(p['hybrid_full_trace_244'] for rec in seeds.values() for p in rec['programs'].values()) for label,seeds in report['models'].items()}
            report['case_comparisons']=len(allcases)
        report['status']='complete'
    except Exception as exc: report.update(status='suspended',error=f'{type(exc).__name__}: {exc}')
    finally:
        try: verify(root);report['protected_unchanged']=True
        except Exception as exc: report.update(status='suspended',protected_unchanged=False,error=f'{type(exc).__name__}: {exc}')
        report['cost']={phase:{k:sum(r[phase][k] for seeds in report['models'].values() for r in seeds.values()) for k in e.KEYS} for phase in ('attempted','completed')}
        expected=dict(zip(e.KEYS,(6,12,12,96))) if qa else BUDGET
        if report['status']=='complete' and any(v!=expected for v in report['cost'].values()):report.update(status='suspended',error='budget mismatch')
        e.write(out/'report.json',report)
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','evaluate','qa']);p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight',type=Path,default=Path('runs/e31_decoded_preflight'));a=p.parse_args()
    r=preflight(a.out) if a.mode=='preflight' else execute(a.out,a.preflight,qa=a.mode=='qa')
    print(json.dumps({'status':r.get('status','preflight_ready'),'cost':r.get('cost')}))
    if r.get('status')=='suspended':raise SystemExit(1)
if __name__=='__main__':main()
