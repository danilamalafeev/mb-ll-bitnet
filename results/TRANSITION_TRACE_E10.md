# E10 transition trace — 6 сентября 2026

Exploratory inference-only trace reusing the E09 manifest (512 paired tables), with fixed hops=2 and recurrent readouts at every step 1–32.

- Source E09 manifest fingerprint: `c0bbe42ce804e6669d6f1dd3eaf69276c664d48f70000867edb0d54d9c3c8c28`; exploratory reuse: `True`.
- Command: `PYTHONPATH=. .venv/bin/python scripts/transition_trace_e10.py`; CPU inference, model.eval, inference_mode; no training.
- Runtime: `2.571` s including prefix checks and traces.

## Counts at step 4k

Counts are predictions equal to f^k(start), out of n=512.

| seed | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 512 | 512 | 192 | 12 | 4 | 7 | 13 | 9 |
| 1 | 512 | 512 | 374 | 70 | 31 | 22 | 20 | 18 |
| 2 | 512 | 512 | 512 | 512 | 506 | 499 | 492 | 481 |

## First-hit summary

For each target f^k(start), first-hit distributions and `never` counts are stored per seed in `report.json`.

| seed | k | first hit median among hits | never |
|---:|---:|---:|---:|
| 0 | 1 | 1 | 0 |
| 0 | 2 | 5 | 0 |
| 0 | 3 | 13 | 17 |
| 0 | 4 | 22 | 182 |
| 0 | 5 | 26 | 419 |
| 0 | 6 | 27 | 471 |
| 0 | 7 | 26 | 479 |
| 0 | 8 | 25 | 488 |
| 1 | 1 | 1 | 0 |
| 1 | 2 | 5 | 0 |
| 1 | 3 | 11 | 9 |
| 1 | 4 | 19 | 159 |
| 1 | 5 | 25 | 326 |
| 1 | 6 | 26 | 415 |
| 1 | 7 | 27 | 446 |
| 1 | 8 | 25 | 459 |
| 2 | 1 | 1 | 0 |
| 2 | 2 | 6 | 0 |
| 2 | 3 | 10 | 0 |
| 2 | 4 | 14 | 0 |
| 2 | 5 | 18 | 3 |
| 2 | 6 | 22 | 6 |
| 2 | 7 | 26 | 14 |
| 2 | 8 | 30 | 25 |

## Interpretation and limits

Seeds0/1 show delayed or unstable transitions after the trained two-hop regime: the full per-step traces and first-hit distributions are the evidence for a possible slowdown. Seed2 remains the comparison checkpoint and first departs from perfect 4k target counts at k=5 (step20: 506/512); its later counts are recorded in report.json. Norms of h, reader updates, and FFN deltas are descriptive only and do not identify a cause. The relative-update summaries are compatible with a possible reduction in state movement as ||h|| grows, but this is only a hypothesis.

All E09 h2 predictions at budgets 4/8/12/16 match exactly for every seed, and ordinary-forward prefix checks at 4/8/12/16/32 pass. This reuses the E09 tables intentionally for exploratory diagnosis; it is not an independent test. Graph distance is decoded from the known cycle, and targets are evaluator-only. First-hit counts can include transient guesses and should not be read as a stable step-by-step algorithm.

## Artifacts

- Source manifest: [`runs/transition_trace_e10/source_manifest.json`](../runs/transition_trace_e10/source_manifest.json)
- Machine-readable report: [`runs/transition_trace_e10/report.json`](../runs/transition_trace_e10/report.json)
