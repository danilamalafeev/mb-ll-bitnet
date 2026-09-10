"""Technical QA only: original failure and one-update temporary patched runner.

Never executes run_experiment. Temporary checkpoints are synthetic QA artifacts,
not scientific u2000 evidence; train_arm's UPDATES is patched to1 for this test.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / 'results/E18_FIRST_ATTEMPT_SOURCE/scripts/step_budget_e18.py'
spec_original = importlib.util.spec_from_file_location('e18_original_snapshot', SOURCE)
original = importlib.util.module_from_spec(spec_original)
spec_original.loader.exec_module(original)
EXPECTED = '31d1143c5194bae6a79f982610bf2baba3fa088a56dcd781aea0f550e6304065'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
text = SOURCE.read_text()
needle = '    PROJECT_ROOT,\n    RUN_CONFIG,'
assert text.count(needle) == 1
fixed = text.replace(needle, '    PROJECT_ROOT,\n    PROGRESS_INTERVAL,\n    RUN_CONFIG,')
assert (ROOT/'scripts/step_budget_e18.py').read_text() == fixed

with tempfile.TemporaryDirectory(prefix='e18_technical_smoke_') as folder:
    tmp = Path(folder)
    candidate = tmp / 'patched_runner.py'
    candidate.write_text(fixed)
    spec = importlib.util.spec_from_file_location('e18_temporary_repair', candidate)
    repaired = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repaired)
    assert repaired.PROGRESS_INTERVAL == 250
    manifest = json.loads((ROOT/'runs/e18_step_budget_preflight/manifest.json').read_text())
    four, eight, _, _ = original.build_paired_models()
    batch = original.fixed_stream()[667][:1]
    assert len(batch) == 1 and len(batch[0].program) == 2
    with patch.object(original, 'UPDATES', 1):
        try:
            original.train_arm(arm='steps4', out=tmp/'original_failure', manifest=manifest,
                               state=four.state_dict(), batches=[batch], root=ROOT)
        except NameError as exc:
            assert 'PROGRESS_INTERVAL' in str(exc)
        else:
            raise AssertionError('original failure was not reproduced')
    assert not list((tmp/'original_failure').glob('*.pt'))
    for arm, model in [('steps4', four), ('steps8', eight)]:
        with patch.object(repaired, 'UPDATES', 1):
            record = repaired.train_arm(arm=arm, out=tmp/arm, manifest=manifest,
                                       state=model.state_dict(), batches=[batch], root=ROOT)
        assert record['status'] == 'complete' and record['update'] == 1
        assert len(record['progress']) == 1 and record['progress'][0]['update'] == 1
        payload = repaired.torch.load(tmp/arm/'final_u2000.pt', weights_only=True)
        assert payload['model_digest'] != manifest['initial_digest']
        assert payload['native_steps'] == {'steps4': 4, 'steps8': 8}[arm]
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
    assert len(original.verify_protected_hashes()) == 138
print(json.dumps({'status': 'PASS', 'original_NameError_reproduced': True,
                  'patched_actual_train_arm': ['steps4', 'steps8'],
                  'tiny_updates_total_including_original_failure': 3,
                  'examples_total': 3, 'internal_substeps_total': 32,
                  'temporary_outputs_removed': True, 'protected_files': 138}))
