# E09 depth transfer — 6 сентября 2026

Inference-only evaluation of existing structured QAT auxiliary checkpoints on 512 fresh, table-disjoint cycle16 test tables. The same table, start, and row order are paired across hops 1–4.

- Manifest: `c0bbe42ce804e6669d6f1dd3eaf69276c664d48f70000867edb0d54d9c3c8c28`; checkpoint seeds: 0, 1, 2; each cell denominator: 512.
- hops4 max encoded length: `55`, model max_tokens: `64`; no evaluator limit change.
- Command: `PYTHONPATH=. .venv/bin/python scripts/depth_transfer_eval.py`.
- Runtime: `4.525` s CPU, including prefix-equivalence checks.

## Main diagonal

Each row is `final_correct / trajectory_correct / n`; the two counts have separate meanings.

| seed | h1@4 | h2@8 | h3@12 | h4@16 |
|---:|---:|---:|---:|---:|
| seed0 | 512 / 512 / 512 | 512 / 512 / 512 | 196 / 196 / 512 | 12 / 12 / 512 |
| seed1 | 512 / 512 / 512 | 512 / 512 / 512 | 378 / 378 / 512 | 73 / 73 / 512 |
| seed2 | 512 / 512 / 512 | 512 / 512 / 512 | 512 / 512 / 512 | 512 / 512 / 512 |

## Full matrix

Trajectory target is f^k(start) at readout step 4k. Every count below is out of n=512.

| seed | hops | budget | final_correct | trajectory_correct | n |
|---:|---:|---:|---:|---:|---:|
| seed0 | 1 | 4 | 512 | 512 | 512 |
| seed0 | 1 | 8 | 0 | 512 | 512 |
| seed0 | 1 | 12 | 0 | 195 | 512 |
| seed0 | 1 | 16 | 0 | 12 | 512 |
| seed0 | 2 | 4 | 0 | 512 | 512 |
| seed0 | 2 | 8 | 512 | 512 | 512 |
| seed0 | 2 | 12 | 320 | 192 | 512 |
| seed0 | 2 | 16 | 32 | 12 | 512 |
| seed0 | 3 | 4 | 0 | 512 | 512 |
| seed0 | 3 | 8 | 0 | 512 | 512 |
| seed0 | 3 | 12 | 196 | 196 | 512 |
| seed0 | 3 | 16 | 456 | 12 | 512 |
| seed0 | 4 | 4 | 0 | 512 | 512 |
| seed0 | 4 | 8 | 0 | 512 | 512 |
| seed0 | 4 | 12 | 0 | 200 | 512 |
| seed0 | 4 | 16 | 12 | 12 | 512 |
| seed1 | 1 | 4 | 512 | 512 | 512 |
| seed1 | 1 | 8 | 0 | 512 | 512 |
| seed1 | 1 | 12 | 0 | 371 | 512 |
| seed1 | 1 | 16 | 4 | 71 | 512 |
| seed1 | 2 | 4 | 0 | 512 | 512 |
| seed1 | 2 | 8 | 512 | 512 | 512 |
| seed1 | 2 | 12 | 137 | 374 | 512 |
| seed1 | 2 | 16 | 10 | 70 | 512 |
| seed1 | 3 | 4 | 0 | 512 | 512 |
| seed1 | 3 | 8 | 0 | 512 | 512 |
| seed1 | 3 | 12 | 378 | 378 | 512 |
| seed1 | 3 | 16 | 401 | 72 | 512 |
| seed1 | 4 | 4 | 0 | 512 | 512 |
| seed1 | 4 | 8 | 0 | 512 | 512 |
| seed1 | 4 | 12 | 0 | 380 | 512 |
| seed1 | 4 | 16 | 73 | 73 | 512 |
| seed2 | 1 | 4 | 512 | 512 | 512 |
| seed2 | 1 | 8 | 0 | 512 | 512 |
| seed2 | 1 | 12 | 0 | 512 | 512 |
| seed2 | 1 | 16 | 0 | 512 | 512 |
| seed2 | 2 | 4 | 0 | 512 | 512 |
| seed2 | 2 | 8 | 512 | 512 | 512 |
| seed2 | 2 | 12 | 0 | 512 | 512 |
| seed2 | 2 | 16 | 0 | 512 | 512 |
| seed2 | 3 | 4 | 0 | 512 | 512 |
| seed2 | 3 | 8 | 0 | 512 | 512 |
| seed2 | 3 | 12 | 512 | 512 | 512 |
| seed2 | 3 | 16 | 0 | 512 | 512 |
| seed2 | 4 | 4 | 0 | 512 | 512 |
| seed2 | 4 | 8 | 0 | 512 | 512 |
| seed2 | 4 | 12 | 0 | 512 | 512 |
| seed2 | 4 | 16 | 512 | 512 | 512 |

## Interpretation and limits

Seeds agree on h1@4 and h2@8. Transfer at h3@12 and h4@16 is seed-dependent: seed2 is perfect on this set, while seeds0/1 are substantially lower. This is evidence of checkpoint dependence, not a pooled claim that all checkpoints transfer.

The paired matrix separates hop-count changes from recurrent-budget changes only partially: changing hops also changes h0 through the STEP count. One set of 512 tables is shared across all three seeds and all four hop strata; these are paired measurements, not 1536 independent tables. Readouts use the shared output head and never feed predictions or targets back into the model. Results are CPU inference-only and do not establish generalization beyond this table distribution or training objective.

## Artifacts

- Manifest: [`runs/depth_transfer_e09/manifest.json`](../runs/depth_transfer_e09/manifest.json)
- Machine-readable report: [`runs/depth_transfer_e09/report.json`](../runs/depth_transfer_e09/report.json)
- Verification: [`runs/depth_transfer_e09/verification.json`](../runs/depth_transfer_e09/verification.json)

## Verification and review

The executor reported **35 passed** for the full suite; the independent reviewer confirmed **4 passed** for the targeted tests and recomputed all 48 cells from the saved predictions. No material blocker was found. The count unit test checks shapes only; the independent artifact recount validates this run. See [independent review](DEPTH_TRANSFER_E09_REVIEW.md).

The manifest retains the source E07 data config with `test_max_hops=3`; E09 explicitly constructs hops1–4, with the actual hops4 length checked against the unchanged model limit.
