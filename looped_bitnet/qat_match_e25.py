"""Fixed E25 QAT masters, provenance, and strict checkpoint format."""
from copy import deepcopy
import json
from pathlib import Path
import torch
from . import longer_native8_e20 as old, replication_e22 as float_reference
from .bit_input_e17 import SignedBitsEncoder
from .register_e15 import QATRegisterModel
from .quantization import BitLinear
from .runtime import seed_everything

ROOT = old.PROJECT_ROOT
SCHEMA = 'e25_qat_match_v1'
SEEDS = (0, 1, 2)
UPDATES = 16000
PREFLIGHT = Path('runs/e25_qat_match_preflight')
RUN = Path('runs/e25_qat_match')
PROTECTED = Path('results/E25_PROTECTED_HASHES.json')
PROTOCOL = Path('results/E25_QAT_MATCH_PROTOCOL.md')
FLOAT_RUN = Path('runs/e24_continuation')
QAT_NAMES = ('reader.q_proj', 'reader.k_proj', 'reader.v_proj', 'reader.out_proj',
             *tuple(f'blocks.{i}.{p}' for i in range(4) for p in ('up', 'down')), 'x_head', 'y_head')
CONFIG = {**float_reference.CONFIG, 'seeds': list(SEEDS), 'updates': UPDATES, 'repeats': 8,
          'quantization': 'existing_BitLinear_ternary_weights_INT8_activations_STE',
          'parameter_count': 151232, 'qat_names': list(QAT_NAMES)}
COST_KEYS = ('program_forwards', 'program_state_cases', 'readout_positions', 'internal_state_substeps')
EVAL_COST = dict(zip(COST_KEYS, (71, 8960, 23488, 187904)))


def check_inventory(model):
    if tuple(n for n, m in model.named_modules() if isinstance(m, BitLinear)) != QAT_NAMES:
        raise ValueError('QAT projection inventory')
    if model.native_steps != 8 or sum(p.numel() for p in model.parameters()) != 151232:
        raise ValueError('QAT model inventory')
    if any(p.dtype != torch.float32 or not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite/non-FP32 master')


def assert_state_equal(left, right, separate=False):
    if list(left) != list(right):
        raise ValueError('state key/order mismatch')
    for key, tensor in left.items():
        other = right[key]
        if tensor.dtype != other.dtype or tensor.shape != other.shape or not torch.equal(tensor, other):
            raise ValueError(f'initial tensor mismatch: {key}')
        if separate and tensor.data_ptr() == other.data_ptr():
            raise ValueError(f'shared storage: {key}')


def build_initial_model(seed):
    if seed not in SEEDS:
        raise ValueError('unregistered seed')
    reference = float_reference.build_initial_model(seed)
    with torch.random.fork_rng(devices=[]):
        seed_everything(seed, deterministic=True, cpu_threads=4)
        model = QATRegisterModel()
        generator = torch.Generator(device='cpu').manual_seed(seed)
        weights = [torch.randn((64, 4), generator=generator, dtype=torch.float32) * .01 for _ in range(2)]
        model.x_embedding = SignedBitsEncoder(weights[0])
        model.y_embedding = SignedBitsEncoder(weights[1])
        model = old.NativeStepRegisterModel.from_e17_bits(model, 8)
        model.initial_rng_state = (torch.Generator(device='cpu').manual_seed(0).get_state()
                                   if seed == 0 else reference.initial_rng_state.clone())
    check_inventory(model)
    assert_state_equal(model.state_dict(), reference.state_dict(), separate=True)
    assert_state_equal(dict(model.named_parameters()), dict(reference.named_parameters()), separate=True)
    return model


def protected(root=ROOT):
    hashes = json.loads((root / PROTECTED).read_text())['sha256']
    if len(hashes) != 249:
        raise ValueError('protected count')
    for path, digest in hashes.items():
        if old.sha256_file(root / path) != digest:
            raise ValueError(f'protected artifact changed: {path}')
    return hashes


def stream():
    base = old.base_stream()
    batches = [list(batch) for _ in range(8) for batch in base]
    for start in range(0, UPDATES, 2000):
        part = batches[start:start + 2000]
        if old.batch_digest(part) != old.E18_BASE_STREAM_DIGEST or old.target_digest(part) != old.E18_BASE_TARGET_DIGEST:
            raise ValueError('repeat stream mismatch')
    return batches


def make_manifest(root=ROOT):
    hashes = protected(root)
    initials, rngs = {}, {}
    for seed in SEEDS:
        model = build_initial_model(seed)
        path = old.E18_INITIAL_STATE if seed == 0 else Path(f'runs/e22_replication_preflight/initial_seed{seed}.pt')
        saved = torch.load(root / path, map_location='cpu', weights_only=True)
        assert_state_equal(model.state_dict(), saved['state_dict'])
        if seed and not torch.equal(model.initial_rng_state, saved['rng_state']):
            raise ValueError('frozen initial RNG mismatch')
        initials[str(seed)] = old.digest_state_dict(model)
        rngs[str(seed)] = old.digest_object(model.initial_rng_state)
    independent = json.loads((root / 'results/E25_INITIAL_REFERENCE.json').read_text())
    for item in independent['seeds']:
        key = str(item['seed'])
        if initials[key] != item['initial_digest'] or rngs[key] != item['float_rng_digest'] or item['bitlinear_locations'] != list(QAT_NAMES):
            raise ValueError('independent initializer reference mismatch')
    if {item['seed'] for item in independent['seeds']} != set(SEEDS):
        raise ValueError('independent seed scope')
    batches = stream()
    previous = json.loads((root / FLOAT_RUN / 'manifest.json').read_text())
    sources = dict(old.E20_SOURCE_RELATIVE_PATHS)
    sources.update({str(p): str(p) for p in (PROTOCOL, Path('looped_bitnet/qat_match_e25.py'),
        Path('scripts/qat_match_e25.py'), Path('tests/test_qat_match_e25.py'),
        Path('looped_bitnet/replication_e22.py'), Path('scripts/replication_e22.py'),
        Path('scripts/composition_e21.py'), Path('scripts/continuation_e24.py'))})
    references = [Path('results/E25_INITIAL_REFERENCE.json'), FLOAT_RUN / 'manifest.json', FLOAT_RUN / 'report.json',
                  *(FLOAT_RUN / f'seed{s}/u16000.pt' for s in SEEDS)]
    return {'schema': SCHEMA, 'config': deepcopy(CONFIG), 'initial_digests': initials,
            'initial_rng_digests': rngs, 'source_hashes': {k: old.sha256_file(root / p) for k,p in sources.items()},
            'protected_snapshot_hash': old.sha256_file(root / PROTECTED), 'protected_count': len(hashes),
            'float_references': {str(p): old.sha256_file(root / p) for p in references},
            'full_stream_digest': old.batch_digest(batches), 'full_target_digest': old.target_digest(batches),
            'base_stream_digest': old.E18_BASE_STREAM_DIGEST, 'base_target_digest': old.E18_BASE_TARGET_DIGEST,
            'checkpoint_costs': {str(UPDATES): {'program_forwards': UPDATES, 'program_state_cases': UPDATES * 64,
                'readout_positions': sum(len(x.program) for b in batches for x in b),
                'internal_state_substeps': 8 * sum(len(x.program) for b in batches for x in b)}},
            'seen': previous['seen'], 'symbolic': previous['symbolic'], 'state_split': previous['state_split']}


def checkpoint_payload(model, optimizer, manifest, seed, update, *, qa=False):
    if seed not in SEEDS or update <= 0 or (not qa and update != UPDATES):
        raise ValueError('checkpoint seed/update')
    check_inventory(model)
    if not model.training:
        raise ValueError('checkpoint must preserve training mode')
    state, opt, rng = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()), torch.get_rng_state().clone()
    return {'schema': SCHEMA, 'seed': seed, 'update': update, 'qa': qa, 'config': deepcopy(CONFIG),
            'completed_updates': update, 'training_cost': manifest['checkpoint_costs'][str(update)],
            'manifest_digest': old.canonical_hash(manifest), 'initial_digest': manifest['initial_digests'][str(seed)],
            'initial_rng_digest': manifest['initial_rng_digests'][str(seed)], 'qat_names': list(QAT_NAMES),
            'state_dict': state, 'model_digest': old.digest_state_dict(model),
            'optimizer_state_dict': opt, 'optimizer_digest': old.digest_object(opt),
            'rng_state': rng, 'rng_digest': old.digest_object(rng), 'training_mode': model.training}


def load_checkpoint(path, manifest, *, seed, update=UPDATES, qa=False):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    with torch.random.fork_rng(devices=[]):
        model = build_initial_model(seed)
        initial = model.state_dict()
        raw = payload['state_dict']
        if list(raw) != list(initial) or any(raw[k].dtype != v.dtype or raw[k].shape != v.shape for k,v in initial.items()):
            raise ValueError('checkpoint tensor inventory')
        model.load_state_dict(raw, strict=True)
        optimizer = old.make_optimizer(model)
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        model.train(payload['training_mode'])
        torch.set_rng_state(payload['rng_state'])
        expected = checkpoint_payload(model, optimizer, manifest, seed, update, qa=qa)
    if set(payload) != set(expected):
        raise ValueError('checkpoint fields')
    for key in expected:
        if key not in ('state_dict', 'optimizer_state_dict', 'rng_state') and payload[key] != expected[key]:
            raise ValueError(f'checkpoint {key} mismatch')
    reference = old.make_optimizer(model).state_dict()['param_groups']
    if optimizer.state_dict()['param_groups'] != reference:
        raise ValueError('optimizer config/membership')
    if len(optimizer.state) != len(list(model.parameters())):
        raise ValueError('optimizer inventory')
    for param, value in optimizer.state.items():
        if set(value) != {'step', 'exp_avg', 'exp_avg_sq'} or int(value['step']) != update:
            raise ValueError('optimizer step/fields')
        for key in ('exp_avg', 'exp_avg_sq'):
            if value[key].shape != param.shape or value[key].dtype != param.dtype or not torch.isfinite(value[key]).all():
                raise ValueError('optimizer tensor inventory')
    torch.set_rng_state(payload['rng_state'])
    return model, optimizer, payload
