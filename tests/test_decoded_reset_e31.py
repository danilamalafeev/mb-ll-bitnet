import copy
import pytest
import torch
from scripts import decoded_reset_e31 as d
from scripts import length_transfer_e28 as e
from tests.test_oracle_reset_e30 import Fake, fresh

def fixture():
    row=copy.deepcopy(d.o.saved()['E27_W4',0][0])
    table=d.o.primitive_table(fresh())
    def outputs():
        return [dict(program=[row['program'][-1]],predictions=[copy.deepcopy(table[(row['program'][-1],*s)]) for s in d.decoded_inputs(row)])]
    return row,table,outputs

def test_own_input_duplicates_and_original_target():
    row,table,outputs=fixture()
    for p in row['predictions'][:2]:
        p['predicted_trace'][3]=[15,15]
        p['prefix_joint_correct'][3]=p['target_trace'][3]==[15,15]
    cases=d.map_cases(row,outputs(),table)
    assert len(cases)==256 and cases[0]['initial_state']!=cases[1]['initial_state']
    assert cases[0]['fresh_input_state']==cases[1]['fresh_input_state']==[15,15]
    assert all(c['fresh_local_correct'] for c in cases)
    assert any(not c['fresh_correct'] and c['fresh_local_target']!=c['target_trace'][-1] for c in cases)
    assert all(c['fresh_correct']==c['fourth_correct'] for c in cases)

@pytest.mark.parametrize('kind',['order','input','prediction','target','flag','missing','program'])
def test_malformed_fresh_rejected(kind):
    row,table,outputs=fixture();rows=outputs();p=rows[0]['predictions'][0]
    if kind=='order': rows[0]['predictions'].reverse()
    elif kind=='input':p['state']=[True,0]
    elif kind=='prediction':p['predicted_trace']=[[16,0]]
    elif kind=='target':p['target_trace']=[[16,0]]
    elif kind=='flag':p['joint_final_correct']=1
    elif kind=='missing':rows[0]['predictions'].pop()
    elif kind=='program':rows[0]['program']=['ADD']

    with pytest.raises(ValueError):d.map_cases(row,rows,table)

@pytest.mark.parametrize('kind',['scope','prediction','target','flag','stratum'])
def test_saved_malformed(kind):
    row,_,_=fixture();p=row['predictions'][0]
    if kind=='scope':row['predictions'].pop()
    elif kind=='prediction':p['predicted_trace'][3]=[True,0]
    elif kind=='target':p['target_trace'][0]=[16,0]
    elif kind=='flag':p['joint_final_correct']=1
    else:p['stratum']='invalid'
    with pytest.raises(ValueError):d.decoded_inputs(row)

def test_strata_and_hybrid_gates():
    cases=[dict(long_final_correct=a,fresh_correct=b,prefix4_all_correct=prefix,fourth_correct=fourth,initial_stratum=s,e30_oracle_correct=True,hybrid_full_trace_correct=prefix and b) for a,b,prefix,fourth,s in [(False,True,True,True,'train'),(True,False,False,False,'test'),(False,True,False,True,'validation')]]
    r=d.summaries(cases)
    assert r['all']['all']['recovered']==2 and r['all']['all']['introduced']==1
    assert r['all']['all']['lost_old_recoveries']==1
    assert r['all']['all']['wrong_prefix_correct_fourth']==1
    assert r['fourth_correct']['all']['denominator']==2 and r['prefix4_correct']['all']['denominator']==1
    assert r['prefix4_wrong']['train']['rescued_fraction'] is None
    gate=d.program_summary([cases[0]]*243+[cases[1]]*13);assert not gate['descriptive_244']
    gate=d.program_summary([cases[0]]*244+[cases[1]]*12);assert gate['descriptive_244'] and gate['hybrid_full_trace_244']
    gate=d.program_summary([cases[2]]*256);assert gate['descriptive_244'] and not gate['hybrid_full_trace_244']

def test_failure_preserved_and_overwrite(tmp_path,monkeypatch):
    def loader(*args):
        m=Fake('exception');return m,torch.optim.AdamW(m.parameters()),{}
    monkeypatch.setattr(e,'load',loader)
    out=tmp_path/'failure';r=d.execute(out,qa=True)
    assert r['status']=='suspended' and r['protected_unchanged']
    assert e.read(out/'report.json')==r and r['cost']['attempted']['program_forwards']==1
    assert r['cost']['completed']['program_forwards']==0
    with pytest.raises(FileExistsError):d.execute(out,qa=True)

def test_preflight_frozen(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'manifest',lambda root:{'version':2})
    out=tmp_path/'run';pre=tmp_path/'pre';pre.mkdir();e.write(pre/'manifest.json',{'version':1})
    with pytest.raises(ValueError,match='preflight changed'):d.execute(out,pre)
    assert not out.exists()


def test_lookup_disagreement_retains_direct_prediction():
    row,table,outputs=fixture();rows=outputs();p=rows[0]['predictions'][0]
    p['predicted_trace'][0][0]^=1;p['joint_final_correct']=False;p['prefix_joint_correct']=[False]
    cases=d.map_cases(row,rows,table)
    assert len(cases)==256 and not cases[0]['direct_lookup_match']
    assert cases[0]['fresh_prediction']==p['predicted_trace'][0]
