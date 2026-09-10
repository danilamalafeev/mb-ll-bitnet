from copy import deepcopy
import json
import pytest
import torch
from looped_bitnet import replication_e22 as e
from looped_bitnet import longer_native8_e20 as old
from looped_bitnet.register_e15 import RegisterExample, evaluate_program, all_programs, forbidden
from scripts import replication_e22 as r
from scripts import composition_e21 as c


def manifest():
    return {'initial_digests': {str(s): old.digest_state_dict(e.build_initial_model(s)) for s in e.SEEDS}}


def test_initialization():
    torch.manual_seed(99); before=torch.get_rng_state().clone()
    models=[e.build_initial_model(s) for s in (0,1,2)]
    assert torch.equal(before,torch.get_rng_state())
    reference=torch.load(e.ROOT/old.E18_INITIAL_STATE,weights_only=True)['state_dict']
    assert all(torch.equal(v,reference[k]) for k,v in models[0].state_dict().items())
    assert len({old.digest_state_dict(m) for m in models})==3
    for seed,model in enumerate(models):
        g=torch.Generator().manual_seed(seed)
        for encoder in (model.x_embedding,model.y_embedding):
            assert torch.equal(encoder.projection.weight,torch.randn((64,4),generator=g)*.01)
        assert old.digest_state_dict(model)==old.digest_state_dict(e.build_initial_model(seed))


def test_actual_runner_roundtrip_and_failed_first_predicate(tmp_path):
    m=manifest(); base=[[RegisterExample(0,1,('ADD',))], [RegisterExample(2,3,('XOR',))]]
    calls=[]
    def evaluator(model,manifest):
        # Actual accepted E21 evaluate/report path, seven legal seen programs.
        programs=[tuple(p) for p in all_programs() if not forbidden(p)][:7]
        result=c._report(c.evaluate_model(model,{'symbolic': {'programs': [], 'targets': {}}},programs=programs),qa=True)
        calls.append(result['forward_passes'])
        return {'primary_conjunction': False, 'combined_conjunction': False, 'seen_prerequisite': False, 'qa_report': result}
    report=r._train(tmp_path/'run',m,_qa=r._TinySpec(base*2,evaluator))
    assert calls==[7,7] and report['status']=='complete' and not report['primary_replication']
    assert set(report['seeds'])=={'1','2'}
    for seed in e.SEEDS:
        model,opt,payload=e.load_checkpoint(tmp_path/'run'/f'seed{seed}'/'u4.pt',m,seed=seed,update=4,qa=True)
        manual=e.build_initial_model(seed); optim=old.make_optimizer(manual)
        for batch in base*2:r._update(manual,optim,batch)
        assert old.digest_object((model.state_dict(),opt.state_dict()))==old.digest_object((manual.state_dict(),optim.state_dict()))
        r._update(model,opt,base[0]); r._update(manual,optim,base[0])
        assert old.digest_object((model.state_dict(),opt.state_dict()))==old.digest_object((manual.state_dict(),optim.state_dict()))
        assert len(report['seeds'][str(seed)]['progress'])==4
        for key,value in [('seed',3),('initial_digest','bad'),('native_steps',4),('model_digest','bad'),('optimizer_digest','bad'),('rng_digest','bad'),('manifest_digest','bad')]:
            corrupt=deepcopy(payload);corrupt[key]=value;path=tmp_path/f'bad{seed}.pt';torch.save(corrupt,path)
            with pytest.raises(ValueError): e.load_checkpoint(path,m,seed=seed,update=4,qa=True)
    with pytest.raises(FileExistsError):r._train(tmp_path/'run',m,_qa=r._TinySpec(base,evaluator))


def test_prerequisite_does_not_replace_primary():
    rows=[{'program':['ADD'],'final_joint':31}, {'program':['ADD','ADD'],'final_joint':31}]
    assert r.predicates({'validation':rows},{'primary_conjunction':True})=={'seen_prerequisite':False,'primary_conjunction':True,'combined_conjunction':False}


def test_incomplete_preserved(tmp_path):
    def broken(*args): raise ValueError('synthetic evaluation failure')
    batch=[[RegisterExample(0,1,('ADD',))]]
    with pytest.raises(ValueError,match='synthetic'):
        r._train(tmp_path/'failed',manifest(),_qa=r._TinySpec(batch,broken))
    progress=json.loads((tmp_path/'failed'/'progress.json').read_text())
    assert progress['status']=='incomplete' and progress['seeds']['2']['status']=='not_started'
    assert (tmp_path/'failed'/'seed1'/'u1.pt').exists()
