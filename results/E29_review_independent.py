"""Independent stdlib-only raw E28 L5 audit; never imports production analyzer."""
import json, hashlib, itertools
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def read(p): return json.loads((root/p).read_text())
ref=read('results/E29_REFERENCE.json')
for p,h in ref['sha256'].items(): assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,p
report=read('runs/e28_length/report.json')
sets={}; summary={}; exposure=Counter(); comps=Counter(); swap=Counter(); features=Counter(); n=0
for family,seeds in report['models'].items():
 for seed,cell in seeds.items():
  label=family+':'+seed; final=set(); trace=set(); recovery=0; seen=set(); prefix=[0]*5
  for row in cell['evaluation']['rows']:
   if len(row['program'])!=5: continue
   rp=[0]*5; rf=rt=rr=0; fd=Counter()
   for case in row['predictions']:
    key=(','.join(row['program']),*case['state']); assert key not in seen;seen.add(key); n+=1
    x,y=case['state']; truth=[]; pres=[]
    for op in row['program']:
     pres.append((x,y))
     if op=='ADD': x=(x+y)%16
     elif op=='XOR': x=x^y
     elif op=='SWAP': x,y=y,x
     else: raise AssertionError(op)
     truth.append([x,y])
    assert truth==case['target_trace']; pred=case['predicted_trace']; ok=[a==b for a,b in zip(truth,pred)]
    assert len(pred)==5 and ok==case['prefix_joint_correct'] and ok[-1]==case['joint_final_correct']
    for i,o in enumerate(ok): rp[i]+=o;prefix[i]+=o
    rf+=ok[-1]; rt+=all(ok);rr+=ok[-1] and not all(ok);fd[str(ok.index(False)+1) if not all(ok) else 'none']+=1
    if not ok[-1]: final.add(key)
    if not all(ok): trace.add(key)
    recovery+=ok[-1] and not all(ok)
    for i,op in enumerate(row['program']):
     if not all(ok[:i]): break
     exposure[(label,i+1,op,'risk')]+=1;exposure[(label,i+1,op,'error')]+=not ok[i]
     x,y=pres[i];fs={'equal':x==y,'zero':x==0 or y==0}
     if op=='ADD':fs.update(overflow=x+y>=16,carry=(x&y)!=0)
     if op=='XOR':fs['popcount']=(x^y).bit_count()
     for f,v in fs.items():
      features[(label,i+1,op,f,str(v),'risk')]+=1;features[(label,i+1,op,f,str(v),'error')]+=not ok[i]
     if not ok[i]:comps[(label,i+1,op,'both' if pred[i][0]!=truth[i][0] and pred[i][1]!=truth[i][1] else 'x_only' if pred[i][0]!=truth[i][0] else 'y_only')]+=1
     if op=='SWAP':swap[(label,i+1,str(x==y),'correct' if ok[i] else 'noop' if pred[i]==[x,y] else 'other')]+=1
   m=row['metrics']['all'];assert rp==m['prefix_joint'] and rf==m['final_joint'] and rt==m['full_trace'] and rr==m['recovery'];assert all(fd[k]==v for k,v in m['first_divergence'].items())
  assert len(seen)==1536
  sets[label]={'final':final,'trace':trace};summary[label]={'final_errors':len(final),'trace_errors':len(trace),'recovery':recovery,'prefix_correct':prefix}
assert n==9216
out={'scope':n,'summary':summary,'exposure':{'|'.join(map(str,k)):v for k,v in exposure.items()},'components':{'|'.join(map(str,k)):v for k,v in comps.items()},'swap':{'|'.join(map(str,k)):v for k,v in swap.items()},'features':{'|'.join(map(str,k)):v for k,v in features.items()},'overlaps':{}}
for typ in ['final','trace']:
 pairs={}
 for a,b in itertools.combinations(sorted(sets),2):
  x,y=sets[a][typ],sets[b][typ];pairs[a+'|'+b]={'intersection':len(x&y),'union':len(x|y)}
 out['overlaps'][typ]=pairs
out['top10']=[{'key':list(k),'final_error_cells':v} for k,v in sorted(Counter(k for c in sets.values() for k in c['final']).items(),key=lambda kv:(-kv[1],kv[0]))[:10]]
(root/'results/E29_review_raw_counts.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out['summary'],indent=2));print('Independent raw audit PASS')
prod=read('results/E29_TRACE_ANALYSIS.json')
for label,ss in summary.items():
 c=label.replace(':','/')
 assert prod['error_counts']['final'][c]==ss['final_errors'] and prod['error_counts']['trace'][c]==ss['trace_errors']
 for r in prod['at_risk'][c]:
  i,op=r['position'],r['opcode'];base=(label,i,op)
  assert r['exposed']==exposure[base+('risk',)] and r['first_errors']==exposure[base+('error',)]
  assert r['rate']==(r['first_errors']/r['exposed'] if r['exposed'] else None)
  for typ,v in r['components'].items(): assert v==comps[base+(typ,)]
  names={'x_equal_y':'equal','either_zero':'zero','overflow':'overflow','carry_present':'carry','xor_popcount':'popcount'}
  for f,bins in r['features'].items():
   for b in bins:
    key=base+(names[f],str(b['value']))
    assert b['exposed']==features[key+('risk',)] and b['first_errors']==features[key+('error',)]
    assert b['rate']==(b['first_errors']/b['exposed'] if b['exposed'] else None)
  if op=='SWAP':
   for eq,st in [('True','equal_inputs'),('False','unequal_inputs')]:
    d=r['swap_patterns'][st]
    for a,b in [('correct_swap','correct'),('wrong_identity','noop'),('other_wrong','other')]:assert d[a]==swap[(label,i,eq,b)]
    assert d['exposed']==sum(d[k] for k in ['correct_swap','wrong_identity','other_wrong'])
for typ in ['final','trace']:
 po=prod['overlaps'][typ]
 for p in po['pairwise']:
  a,b=[c.replace('/',':') for c in p['cells']];x,y=sets[a][typ],sets[b][typ]
  assert p['intersection']==len(x&y) and p['union']==len(x|y) and p['a_only']==len(x-y) and p['b_only']==len(y-x)
  assert p['jaccard']==(len(x&y)/len(x|y) if x|y else None)
 for family in report['models']:
  a,b,c=[sets[family+':'+str(s)][typ] for s in range(3)]
  d=po['within_family'][family];assert d['intersection']==len(a&b&c) and d['union']==len(a|b|c)
  d=po['seed1_contrast'][family];assert d['common_seed0_seed2']==len(a&c) and d['common_corrected_by_seed1']==len((a&c)-b) and d['seed1_unique']==len(b-(a|c)) and d['seed1_errors']==len(b)
 allsets=[s[typ] for s in sets.values()];assert po['all_six']['intersection']==len(set.intersection(*allsets)) and po['all_six']['union']==len(set.union(*allsets))
assert [(e['key'],e['final_error_cells']) for e in out['top10']]==[([','.join(e['program']),*e['state']],len(e['final_error_cells'])) for e in prod['examples']]
print('Production comparison PASS: all cell exposures/features/components/SWAP, overlaps, contrasts, top10')
for row in prod['at_risk']['all_cells']:
 local=[next(r for r in prod['at_risk'][c] if (r['position'],r['opcode'])==(row['position'],row['opcode'])) for c in prod['summary']]
 for f in ['observations','exposed','first_errors']:assert row[f]==sum(r[f] for r in local)
 for f,v in row['components'].items():assert v==sum(r['components'][f] for r in local)
 for f,bins in row['features'].items():
  for b in bins:
   lbs=[next(z for z in r['features'][f] if z['value']==b['value']) for r in local]
   for key in ['exposed','first_errors']:assert b[key]==sum(z[key] for z in lbs)
 if 'swap_patterns' in row:
  for group,d in row['swap_patterns'].items():
   for f,v in d.items():assert v==sum(r['swap_patterns'][group][f] for r in local)
print('Pooled table additive check PASS')
