"""E22 initialization-only replication; fixed data and final budget."""
from copy import deepcopy
from pathlib import Path
import torch
from . import longer_native8_e20 as e20
from .bit_input_e17 import SignedBitsEncoder
from .float_qat_e16 import replace_all_bitlinear
from .register_e15 import QATRegisterModel
from .runtime import seed_everything

ROOT = e20.PROJECT_ROOT
SCHEMA = 'e22_replication_v1'
SEEDS = (1, 2)
UPDATES = 8000
PREFLIGHT = Path('runs/e22_replication_preflight')
RUN = Path('runs/e22_replication')
PROTECTED = Path('results/E22_PROTECTED_HASHES.json')
CONFIG = {'seeds': [1, 2], 'data_seed': 0, 'projection_seed': 'initialization_seed',
          'updates': 8000, 'base_updates': 2000, 'repeats': 4, 'native_steps': 8,
          'batch_size': 64, 'optimizer': e20.OPTIMIZER_CONFIG, 'cpu_threads': 4,
          'precision': 'float32', 'autocast': False, 'clip_grad_norm': 1.0}
TRAIN_COST = {'updates': 16000, 'examples': 1024000, 'internal_state_substeps': 16384000}
EVAL_COST = {'program_state_cases': 17920, 'readout_positions': 46976, 'internal_state_substeps': 375808}


def build_initial_model(seed):
    if seed not in (0, 1, 2):
        raise ValueError('unregistered initialization seed')
    with torch.random.fork_rng(devices=[]):
        seed_everything(seed, deterministic=True, cpu_threads=4)
        model, _ = replace_all_bitlinear(QATRegisterModel())
        generator = torch.Generator(device='cpu').manual_seed(seed)
        weights = [torch.randn((64, 4), generator=generator, dtype=torch.float32) * .01 for _ in range(2)]
        model.x_embedding = SignedBitsEncoder(weights[0])
        model.y_embedding = SignedBitsEncoder(weights[1])
        model = e20.NativeStepRegisterModel.from_e17_bits(model, 8)
        model.initial_rng_state = torch.get_rng_state().clone()
    if sum(p.numel() for p in model.parameters()) != 151232 or any(m.__class__.__name__ == 'BitLinear' for m in model.modules()):
        raise ValueError('initial inventory mismatch')
    return model


def checkpoint_payload(model, optimizer, manifest, seed, update, *, qa=False):
    if seed not in SEEDS or update <= 0 or (not qa and update != UPDATES):
        raise ValueError('checkpoint seed/update')
    if model.native_steps != 8 or sum(p.numel() for p in model.parameters()) != 151232:
        raise ValueError('checkpoint inventory')
    state = deepcopy(model.state_dict()); opt = deepcopy(optimizer.state_dict()); rng = torch.get_rng_state().clone()
    return {'schema': SCHEMA, 'seed': seed, 'update': update, 'qa': qa, 'native_steps': 8,
            'encoder_mode': 'bits', 'parameter_count': 151232, 'manifest_digest': e20.canonical_hash(manifest),
            'initial_digest': manifest['initial_digests'][str(seed)], 'config': CONFIG,
            'state_dict': state, 'model_digest': e20.digest_state_dict(model),
            'optimizer_state_dict': opt, 'optimizer_digest': e20.digest_object(opt),
            'rng_state': rng, 'rng_digest': e20.digest_object(rng),
            'training_mode': model.training}


def load_checkpoint(path, manifest, *, seed, update=UPDATES, qa=False):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    with torch.random.fork_rng(devices=[]):
        model = build_initial_model(seed); optimizer = e20.make_optimizer(model)
        initial = model.state_dict()
        if list(payload['state_dict']) != list(initial) or any(payload['state_dict'][k].dtype != initial[k].dtype or payload['state_dict'][k].shape != initial[k].shape for k in initial):
            raise ValueError('checkpoint tensor inventory')
        model.load_state_dict(payload['state_dict'], strict=True)
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        torch.set_rng_state(payload['rng_state'])
        expected = checkpoint_payload(model, optimizer, manifest, seed, update, qa=qa)
    if set(payload) != set(expected):
        raise ValueError('checkpoint fields')
    for key in expected:
        if key not in ('state_dict', 'optimizer_state_dict', 'rng_state') and payload[key] != expected[key]:
            raise ValueError(f'checkpoint {key} mismatch')
    reference = e20.make_optimizer(model).param_groups[0]
    for group in optimizer.param_groups:
        if any(group[k] != reference[k] for k in ('lr', 'weight_decay', 'betas', 'eps', 'amsgrad', 'foreach')):
            raise ValueError('optimizer configuration')
    if len(optimizer.state) != len(list(model.parameters())) or any(int(v['step']) != update for v in optimizer.state.values()):
        raise ValueError('optimizer steps/inventory')
    return model, optimizer, payload
