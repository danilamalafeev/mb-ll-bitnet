# E12 state-scale schedule — 6 сентября 2026

Inference-only comparison of one centered-RMS reset with repeated resets on a new 512-table cycle16 test manifest.

- Manifest fingerprint: `3f615c6324a0044dc3dee485d0e91377c014b087eb1ec8e2b8db8f014124f530`; stream seed `20260909`; novelty overlap counts: `{'e07_eval_sets': 0, 'e07_reconstructed_suites': 0, 'e08_pairs': 0, 'e09_manifest': 0, 'e10_manifest': 0, 'e11_manifest': 0}`.
- Command: `PYTHONPATH=. .venv/bin/python scripts/state_scale_schedule_e12.py`; CPU, batch `64`, fixed hops `2`, 32 steps; runtime including controls `5.118` s.
- Schedules: baseline `()`, noop `()`, single `(8,)`, repeated `(8, 12, 16, 20, 24, 28)`; epsilon `1e-08`.

## Counts at step 4k

- seed0: baseline `{'1': 512, '2': 512, '3': 196, '4': 13, '5': 10, '6': 8, '7': 15, '8': 11}`; noop `{'1': 512, '2': 512, '3': 196, '4': 13, '5': 10, '6': 8, '7': 15, '8': 11}`; single `{'1': 512, '2': 512, '3': 459, '4': 322, '5': 121, '6': 41, '7': 21, '8': 18}`; repeated `{'1': 512, '2': 512, '3': 459, '4': 363, '5': 291, '6': 238, '7': 213, '8': 178}`; repeated−single k8 `160`; paired wins/losses `170/10`
- seed1: baseline `{'1': 512, '2': 512, '3': 388, '4': 75, '5': 40, '6': 17, '7': 19, '8': 17}`; noop `{'1': 512, '2': 512, '3': 388, '4': 75, '5': 40, '6': 17, '7': 19, '8': 17}`; single `{'1': 512, '2': 512, '3': 506, '4': 459, '5': 247, '6': 109, '7': 60, '8': 43}`; repeated `{'1': 512, '2': 512, '3': 506, '4': 481, '5': 451, '6': 419, '7': 399, '8': 371}`; repeated−single k8 `328`; paired wins/losses `331/3`
- seed2: baseline `{'1': 512, '2': 512, '3': 512, '4': 511, '5': 505, '6': 504, '7': 500, '8': 488}`; noop `{'1': 512, '2': 512, '3': 512, '4': 511, '5': 505, '6': 504, '7': 500, '8': 488}`; single `{'1': 512, '2': 512, '3': 512, '4': 512, '5': 508, '6': 507, '7': 504, '8': 496}`; repeated `{'1': 512, '2': 512, '3': 512, '4': 512, '5': 512, '6': 511, '7': 511, '8': 509}`; repeated−single k8 `13`; paired wins/losses `13/0`

## Predeclared criterion

Practical usefulness requires repeated−single k8 ≥ 26/512 for both seeds 0 and 1. Seed2 is displayed independently; this threshold is diagnostic, not a significance test.

## Controls and artifacts

{'prefix_checks': True, 'noop_exact': True, 'single_repeated_identical_through12': True, 'checkpoint_hashes_unchanged': True, 'novelty_zero_all_scopes': True}

Per-example predictions, r4, pre/post RMS, factors, checkpoint hashes, manifests and full configuration are in `report.json`.
