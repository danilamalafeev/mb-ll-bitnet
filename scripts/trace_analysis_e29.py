"""Deterministic, stdlib-only descriptive analysis of frozen E28 saved traces."""
import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path

FAMILIES = ('E24_float', 'E27_W4')
STATES = {(x, y) for x in range(16) for y in range(16)}
OPS = ('ADD', 'XOR', 'SWAP')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def read(path):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            require(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    return json.loads(path.read_text(), object_pairs_hook=unique)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pair(value):
    require(isinstance(value, list) and len(value) == 2 and all(type(x) is int and 0 <= x < 16 for x in value), 'malformed register pair')
    return tuple(value)

def trace(program, state):
    x, y = state
    result = []
    for op in program:
        require(op in OPS, 'unknown opcode')
        if op == 'ADD': x = (x + y) % 16
        elif op == 'XOR': x = x ^ y
        else: x, y = y, x
        result.append([x, y])
    return result

def index(items, key):
    out = {}
    for item in items:
        k = key(item)
        require(k not in out, 'duplicate identity')
        out[k] = item
    return out

def diagnose(program, item, membership):
    state = pair(item['state'])
    expected = trace(program, state)
    require(item['target_trace'] == expected, 'target mismatch')
    pred = item['predicted_trace']
    require(isinstance(pred, list) and len(pred) == len(program), 'trace length')
    for p in pred: pair(p)
    correct = [a == b for a, b in zip(expected, pred)]
    require(item['prefix_joint_correct'] == correct and all(type(x) is bool for x in item['prefix_joint_correct']), 'prefix flags')
    require(type(item['joint_final_correct']) is bool and item['joint_final_correct'] == correct[-1], 'final flag')
    require(item['stratum'] == membership[state], 'stratum mismatch')
    first = next((i for i, c in enumerate(correct) if not c), None)
    return dict(state=list(state), expected=expected, predicted=pred, correct=correct,
                first=first, recovery=first is not None and correct[-1], stratum=membership[state])

def metrics(records, length=5):
    first = {str(i+1): sum(r['first'] == i for r in records) for i in range(length)}
    first['none'] = sum(r['first'] is None for r in records)
    return dict(denominator=len(records), final_joint=sum(r['correct'][-1] for r in records),
                full_trace=sum(all(r['correct']) for r in records),
                prefix_joint=[sum(r['correct'][i] for r in records) for i in range(length)],
                final_x=sum(r['expected'][-1][0] == r['predicted'][-1][0] for r in records),
                final_y=sum(r['expected'][-1][1] == r['predicted'][-1][1] for r in records),
                first_divergence=first, recovery=sum(r['recovery'] for r in records))

def overlap(sets):
    union = set.union(*sets)
    common = set.intersection(*sets)
    return dict(intersection=len(common), union=len(union), jaccard=len(common)/len(union) if union else None)

def features(op, pre):
    x, y = pre
    out = {'x_equal_y': x == y, 'either_zero': x == 0 or y == 0}
    if op == 'ADD': out.update(overflow=x+y >= 16, carry_present=(x & y) != 0)
    if op == 'XOR': out['xor_popcount'] = (x ^ y).bit_count()
    return out

def risk_table(records):
    rows = []
    for i in range(5):
        for op in OPS:
            observed = [(p,r) for p,r in records if p[i] == op]
            risk = [(p,r) for p,r in observed if all(r['correct'][:i])]
            errors = [(p,r) for p,r in risk if not r['correct'][i]]
            row = dict(position=i+1, opcode=op, observations=len(observed), exposed=len(risk), first_errors=len(errors), rate=len(errors)/len(risk) if risk else None)
            row['components'] = dict.fromkeys(('x_only','y_only','both'),0)
            for _,r in errors:
                bad = [a != b for a,b in zip(r['expected'][i],r['predicted'][i])]
                row['components']['both' if all(bad) else 'x_only' if bad[0] else 'y_only'] += 1
            names = {'x_equal_y': [False,True], 'either_zero':[False,True]}
            if op == 'ADD': names.update(overflow=[False,True],carry_present=[False,True])
            if op == 'XOR': names['xor_popcount'] = list(range(5))
            row['features'] = {}
            for name, values in names.items():
                bins = []
                for value in values:
                    subset = [r for _,r in risk if features(op, r['state'] if i == 0 else r['expected'][i-1])[name] == value]
                    n = sum(not r['correct'][i] for r in subset)
                    bins.append(dict(value=value,exposed=len(subset),first_errors=n,rate=n/len(subset) if subset else None))
                row['features'][name] = bins
            if op == 'SWAP':
                row['swap_patterns'] = {}
                for same in (False,True):
                    counts = dict(exposed=0,correct_swap=0,wrong_identity=0,other_wrong=0)
                    for _,r in risk:
                        pre = r['state'] if i == 0 else r['expected'][i-1]
                        if (pre[0] == pre[1]) != same: continue
                        counts['exposed'] += 1
                        category = 'correct_swap' if r['correct'][i] else 'wrong_identity' if r['predicted'][i] == pre else 'other_wrong'
                        counts[category] += 1
                    row['swap_patterns']['equal_inputs' if same else 'unequal_inputs'] = counts
            rows.append(row)
    return rows

def analyze(root):
    reference = read(root/'results/E29_REFERENCE.json')
    frozen = dict(reference['sha256'])
    manifest = read(root/'runs/e28_length/manifest.json')
    frozen.update(manifest['protected'])
    frozen.update(manifest['source_hashes'])
    frozen['results/E29_REFERENCE.json'] = digest(root/'results/E29_REFERENCE.json')
    for path, sha in frozen.items(): require(digest(root/path) == sha, 'input hash mismatch: '+path)
    selection = read(root/'results/E28_SELECTION.json')
    require(manifest['selection'] == selection, 'manifest selection')
    require(manifest['seeds'] == [0,1,2] and set(manifest['models']) == set(FAMILIES), 'manifest cells')
    strata_path = 'runs/register_e15_preflight/v7/manifest.json'
    strata = read(root/strata_path)['state_split']
    require({k:len(v) for k,v in strata.items()} == dict(train=192,validation=32,test=32), 'strata sizes')
    membership = {}
    for name, states in strata.items():
        for s in states:
            s = pair(s); require(s not in membership, 'overlapping strata'); membership[s] = name
    require(set(membership) == STATES, 'strata scope')
    frozen[strata_path] = digest(root/strata_path)
    selected = index([p for p in selection['programs'] if p['length']==5], lambda p:tuple(p['program']))
    require(len(selected)==6 and all(len(p)==5 for p in selected), 'L5 program scope')
    for p,s in selected.items():
        cases = index(s['cases'], lambda c:pair(c['state']))
        require(set(cases)==STATES, 'selection state scope')
        for state,c in cases.items(): require(c['target_trace']==trace(p,state), 'selection DSL')
    report = read(root/'runs/e28_length/report.json')
    require(report['status']=='complete' and report['qa'] is False and report['training_updates']==0, 'report status')
    require(set(report['models'])==set(FAMILIES), 'report families')
    cells = {}; summaries = {}
    for family in FAMILIES:
        require(set(report['models'][family])=={'0','1','2'}, 'seed scope')
        for seed, cell in sorted(report['models'][family].items()):
            cid = family+'/'+seed
            require(cell['unchanged'] is True, 'model changed')
            require(cell['attempted']=={k:v//6 for k,v in manifest['budget'].items()}, 'saved work counts')
            require(cell['completed']==cell['attempted'], 'incomplete work')
            partial = index(cell['partial_rows'], lambda r:tuple(r['program']))
            evaluated = index(cell['evaluation']['rows'], lambda r:tuple(r['program']))
            require(set(partial)==set(evaluated)=={tuple(p['program']) for p in selection['programs']}, 'row scope')
            cells[cid] = {}; summaries[cid] = {}
            for program,s in sorted(selected.items()):
                row = partial[program]; ev = evaluated[program]
                require(ev['name']==s['name'] and ev['length']==5, 'program identity')
                preds = index(row['predictions'], lambda r:pair(r['state']))
                epreds = index(ev['predictions'], lambda r:pair(r['state']))
                require(set(preds)==STATES and preds==epreds, 'prediction identity')
                records = [diagnose(program,preds[state],membership) for state in sorted(STATES)]
                for i in range(5):
                    require({tuple(r['state'] if i==0 else r['expected'][i-1]) for r in records} == STATES, 'true prestate bijection')
                m = metrics(records)
                for a,b in [('states','denominator'),('joint_final','final_joint'),('full_trace','full_trace'),('prefix_joint','prefix_joint'),('final_x_correct','final_x'),('final_y_correct','final_y')]: require(row[a]==m[b], 'partial counter '+a)
                for stratum in ('all','train','validation','test'):
                    subset = records if stratum=='all' else [r for r in records if r['stratum']==stratum]
                    require(ev['metrics'][stratum]==metrics(subset), 'saved metrics mismatch')
                summaries[cid][s['name']] = m
                for r in records: cells[cid][(program,*r['state'])] = r
    require(sum(len(v) for v in cells.values())==9216, 'trace count')
    require(report['cost']['attempted']==report['cost']['completed']==manifest['budget'], 'aggregate saved work')
    require(set(report['paired'])=={'0','1','2'}, 'paired seeds')
    for seed in ('0','1','2'):
        paired = index(report['paired'][seed], lambda r:r['name'])
        for program,s in selected.items():
            a=cells['E27_W4/'+seed]; b=cells['E24_float/'+seed]
            for stratum in ('all','train','validation','test'):
                keys=[k for k in a if k[0]==program and (stratum=='all' or a[k]['stratum']==stratum)]
                for metric in ('final_joint','full_trace'):
                    counts=dict(W4_wins=0,float_wins=0,both_correct=0,both_wrong=0)
                    for k in keys:
                        ac=a[k]['correct'][-1] if metric=='final_joint' else all(a[k]['correct'])
                        bc=b[k]['correct'][-1] if metric=='final_joint' else all(b[k]['correct'])
                        counts['both_correct' if ac and bc else 'both_wrong' if not ac and not bc else 'W4_wins' if ac else 'float_wins']+=1
                    require(paired[s['name']]['metrics'][stratum][metric]==counts,'saved paired counters')
    errors = {kind:{c:{k for k,r in rs.items() if not (r['correct'][-1] if kind=='final' else all(r['correct']))} for c,rs in cells.items()} for kind in ('final','trace')}
    overlaps = {}
    for kind, sets in errors.items():
        pairs = [dict(cells=[a,b],a_errors=len(sets[a]),b_errors=len(sets[b]),a_only=len(sets[a]-sets[b]),b_only=len(sets[b]-sets[a]),**overlap([sets[a],sets[b]])) for a,b in combinations(cells,2)]
        families = {f:overlap([sets[f+'/'+str(s)] for s in range(3)]) for f in FAMILIES}
        contrast = {}
        for f in FAMILIES:
            a,b,c = [sets[f+'/'+str(s)] for s in range(3)]
            contrast[f] = dict(common_seed0_seed2=len(a&c),common_corrected_by_seed1=len((a&c)-b),seed1_errors=len(b),seed1_unique=len(b-(a|c)),seed0_or_seed2_errors=len(a|c),case_universe=1536)
        overlaps[kind] = dict(pairwise=pairs,within_family=families,all_six=overlap(list(sets.values())),seed1_contrast=contrast,
            paired_same_seed=[dict(seed=s,both_wrong=len(sets['E27_W4/'+str(s)]&sets['E24_float/'+str(s)]),W4_only_wrong=len(sets['E27_W4/'+str(s)]-sets['E24_float/'+str(s)]),float_only_wrong=len(sets['E24_float/'+str(s)]-sets['E27_W4/'+str(s)]),both_correct=1536-len(sets['E27_W4/'+str(s)]|sets['E24_float/'+str(s)])) for s in range(3)])
    tables = {c:risk_table([(k[0],r) for k,r in rs.items()]) for c,rs in cells.items()}
    tables['all_cells'] = risk_table([(k[0],r) for rs in cells.values() for k,r in rs.items()])
    ranked = sorted(set.union(*errors['final'].values()),key=lambda k:(-sum(k in e for e in errors['final'].values()),k))[:10]
    examples = []
    for key in ranked:
        entry = dict(program=list(key[0]),state=list(key[1:]),final_error_cells=[c for c,e in errors['final'].items() if key in e],cells={})
        for c,rs in cells.items():
            r=rs[key]; i=r['first']
            entry['cells'][c] = dict(final_error=not r['correct'][-1],first_error_position=None if i is None else i+1,opcode=None if i is None else key[0][i],true_prestate=None if i is None else r['state'] if i==0 else r['expected'][i-1],expected=None if i is None else r['expected'][i],predicted=None if i is None else r['predicted'][i],recovery=r['recovery'])
        examples.append(entry)
    for path,sha in frozen.items(): require(digest(root/path)==sha,'input changed: '+path)
    return dict(schema='e29_saved_trace_analysis_v1',input_hashes=frozen,accounting=dict(saved_case_traces=9216,instruction_observations=46080,training_updates=0,model_forwards=0,checkpoint_loads=0),summary=summaries,error_counts={k:{c:len(s) for c,s in v.items()} for k,v in errors.items()},overlaps=overlaps,at_risk=tables,examples=examples,limitations=['Exploratory descriptive saved-output analysis; no causal inference or independent-trial assumption.','Length and selected composition remain confounded; no hidden-state inference from decoded readouts.'])

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); parser.add_argument('--output',type=Path,default=Path('results/E29_TRACE_ANALYSIS.json'))
    args=parser.parse_args(); result=analyze(args.root)
    path=args.output if args.output.is_absolute() else args.root/args.output
    with path.open('x') as f: json.dump(result,f,indent=2,sort_keys=True); f.write('\n')
    print(json.dumps(result['error_counts'],sort_keys=True))

if __name__=='__main__': main()
