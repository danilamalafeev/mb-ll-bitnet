# E11 state-scale intervention — 6 сентября 2026

Inference-only centered-RMS intervention on a new 512-table cycle16 test manifest, fixed hops=2 and 32 recurrent steps.

- Manifest fingerprint: `98ce79f0db6d81bb379fc80d419cd3f59f72758a76c76f30c286d28b3b99c68a`; stream seed `20260908`; zero overlap with E09/E07/E08: `True`.
- Command: `PYTHONPATH=. .venv/bin/python scripts/state_scale_e11.py`; CPU, model.eval, inference_mode, no training.
- Runtime: `5.293` s; interventions at steps (8, 12, 16, 20, 24, 28); epsilon `1e-8`.

## Counts at step 4k

Each value is correct predictions of f^k(start) out of n=512.

- seed0: baseline `{'1': 512, '2': 512, '3': 195, '4': 15, '5': 12, '6': 13, '7': 12, '8': 7}`; noop `{'1': 512, '2': 512, '3': 195, '4': 15, '5': 12, '6': 13, '7': 12, '8': 7}`; intervention `{'1': 512, '2': 512, '3': 468, '4': 384, '5': 308, '6': 250, '7': 220, '8': 189}`
- seed1: baseline `{'1': 512, '2': 512, '3': 392, '4': 79, '5': 28, '6': 17, '7': 12, '8': 21}`; noop `{'1': 512, '2': 512, '3': 392, '4': 79, '5': 28, '6': 17, '7': 12, '8': 21}`; intervention `{'1': 512, '2': 512, '3': 507, '4': 487, '5': 453, '6': 430, '7': 395, '8': 365}`
- seed2: baseline `{'1': 512, '2': 512, '3': 512, '4': 510, '5': 506, '6': 501, '7': 499, '8': 492}`; noop `{'1': 512, '2': 512, '3': 512, '4': 510, '5': 506, '6': 501, '7': 499, '8': 492}`; intervention `{'1': 512, '2': 512, '3': 512, '4': 512, '5': 510, '6': 510, '7': 510, '8': 509}`

## Checks and interpretation

No-op is an explicit identity branch and must reproduce baseline. The centered-RMS intervention changes state scale while preserving the per-example mean and centered direction. Any improvement would show that this intervention changes behavior under this diagnostic, not that scale is the unique natural cause of the E09 errors.

Per-example predictions, factors, pre/post intervention logit differences, argmax matches, norms, checkpoint hashes, and full config are in `report.json`. Targets are evaluator-only. The new manifest is independent of E09/E07/E08 by the recorded overlap check.

## Artifacts

- Manifest: [`runs/state_scale_e11/manifest.json`](../runs/state_scale_e11/manifest.json)
- Report: [`runs/state_scale_e11/report.json`](../runs/state_scale_e11/report.json)
