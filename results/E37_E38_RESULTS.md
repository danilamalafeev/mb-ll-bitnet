# E37/E38 saved-results report

**Status: ACCEPT (independent raw audit and strict replay).** This report was assembled from the saved JSON reports after the single detached science run. It did not load checkpoints, construct models, run forwards, train, or rerun preflight/QA.

The detached supervisor finished with exit code 0 and queued one wake message. Its recorded wall interval was `2026-09-08T16:05:07Z` to `2026-09-08T18:16:57Z` (7,910 seconds). The runner reports `complete` for the combined E37/E38 output, both subexperiments, all four E38 parent extensions, and all eight E38 A/B trajectories. No numerical or technical failure was recorded. The independent raw audit and strict replay both passed, so the artifact gate is ACCEPT.

The frozen E37/E38 base manifest is `runs/e37_e38_preflight_v2/manifest.json` (SHA256 `593cd1fb98ee180b13f968060a329d11936f61f514a45517b97d8227ffddb9b2`; canonical digest `42bd72d9a4f2e7119282278c9e96eb40b888117eb1f4b59de63076b4dff5a956`). The E37 and E38 output manifests have the same byte hash. E38 records four append-only parent registrations in `runs/e37_e38_science_v3/e38/runtime_lineage.json` under schema `e37_e38_runtime_lineage_v2`; every parent has status `complete` with 16,000 attempted and completed updates. The source/reference binding is `results/E37_E38_SCIENCE_REFERENCE_V3.json` (SHA256 `d02e663f717953b224f56b2081dca5269909612bd89afcd8b7e945a7df4d8dd0`).

## Accounting

E37 is inference-only: 216 program forwards, 55,296 state cases, 331,776 readout positions, and 2,654,208 native internal steps. E38 uses four own-parent extensions of 16,000 updates and eight branch trajectories of 8,000 updates. Its training total is 128,000 updates, 8,192,000 examples, 19,456,000 readout positions, and 155,648,000 native internal steps. E38 evaluation totals are 3,080 forwards, 730,880 state cases, 4,341,760 readout positions, and 34,734,080 native internal steps.

Combined scientific evaluation is therefore 3,296 forwards, 786,176 state cases, 4,673,536 readout positions, and 37,388,288 native internal steps. Combined training remains 128,000 updates, 8,192,000 examples, 19,456,000 readout positions, and 155,648,000 native internal steps. The complete machine-readable accounting is in [E37_E38_ACCOUNTING.json](E37_E38_ACCOUNTING.json).

## E37 fixed padding

Each cell is `final errors / full-trace errors` over 2,304 cases for that padded length (9 programs × 256 states). Every endpoint contains 36 programs and 9,216 cases.

| precision / branch | L3 | L5 | L7 | L9 |
|---|---:|---:|---:|---:|
| float / A | 0 / 0 | 13 / 13 | 599 / 645 | 1,661 / 1,858 |
| float / B@4572 | 0 / 0 | 0 / 0 | 0 / 0 | 73 / 73 |
| float / B8000 | 0 / 0 | 0 / 0 | 0 / 0 | 56 / 60 |
| W4 / A | 1 / 1 | 70 / 71 | 1,264 / 1,339 | 2,093 / 2,235 |
| W4 / B@4572 | 0 / 0 | 0 / 0 | 1 / 1 | 207 / 217 |
| W4 / B8000 | 0 / 0 | 0 / 0 | 1 / 1 | 155 / 161 |

For paired A-versus-branch comparisons, `introduced` is A-correct/B-wrong and `recovered` is A-wrong/B-correct. Counts are over the 9,216 identical program/state pairs; `both wrong` is also retained.

| precision / comparison | final introduced / recovered / both wrong | full-trace introduced / recovered / both wrong |
|---|---:|---:|
| float A vs B@4572 | 9 / 2,209 / 64 | 4 / 2,447 / 69 |
| float A vs B8000 | 7 / 2,224 / 49 | 1 / 2,457 / 59 |
| W4 A vs B@4572 | 14 / 3,234 / 194 | 3 / 3,431 / 215 |
| W4 A vs B8000 | 16 / 3,288 / 140 | 1 / 3,485 / 161 |

The paired result is descriptive: padding exposes a degradation pattern for A, while the B endpoints recover many of those paired cases. It is not a claim about an isolated mechanism.

## E38 own parents

Each parent endpoint was evaluated once with 83 forwards, 12,032 cases, 37,312 readout positions, and 298,496 native internal steps. The table gives final/full-trace errors over six programs and 1,536 program/state cases at each length.

| parent | L4 | L5 | recorded predicates |
|---|---:|---:|---|
| float seed1 | 9 / 9 | 92 / 96 | seen and E21 primary true; L4 true; L5 false |
| float seed2 | 2 / 2 | 100 / 102 | seen and E21 primary true; L4 true; L5 false |
| W4 seed1 | 21 / 23 | 202 / 211 | seen and E21 primary true; L4 false; L5 false |
| W4 seed2 | 5 / 5 | 97 / 103 | seen and E21 primary true; L4 true; L5 false |

These are descriptive prerequisite results. They did not select or exclude any parent from E38.

## E38 primary held-out result

The primary pool is the registered common allowed semantic-heldout L7–L10 pool: 24 programs and 6,144 cases per endpoint. Cells show `final errors / full-trace errors`; the four length entries are L7, L8, L9, L10, followed by the total.

| model | A@8000 | B@4572 | B@8000 |
|---|---|---|---|
| float seed1 | 437/446, 962/999, 1,120/1,150, 1,394/1,429; **3,913/4,024** | 0/0, 1/1, 1/1, 21/23; **23/25** | 0/0, 0/0, 0/0, 12/12; **12/12** |
| float seed2 | 440/454, 936/985, 1,188/1,229, 1,426/1,456; **3,990/4,124** | 0/0, 0/0, 1/1, 9/9; **10/10** | 0/0, 0/0, 1/1, 8/8; **9/9** |
| W4 seed1 | 524/547, 1,080/1,135, 1,170/1,236, 1,414/1,472; **4,188/4,390** | 0/0, 3/3, 3/3, 23/25; **29/31** | 0/0, 0/0, 0/0, 15/16; **15/16** |
| W4 seed2 | 298/308, 809/855, 938/995, 1,333/1,380; **3,378/3,538** | 0/0, 0/0, 2/2, 11/16; **13/18** | 0/0, 0/0, 3/3, 7/9; **10/12** |

The paired directional counts are also saved for every model. In the order `A-correct/B-wrong` versus `A-wrong/B-correct`, final/full-trace counts were: float seed1 B@4572 `0/3,890` and `0/3,999`, B8000 `0/3,901` and `0/4,012`; float seed2 B@4572 `1/3,981` and `0/4,114`, B8000 `0/3,981` and `0/4,115`; W4 seed1 B@4572 `5/4,164` and `2/4,361`, B8000 `0/4,173` and `0/4,374`; W4 seed2 B@4572 `0/3,365` and `0/3,520`, B8000 `0/3,368` and `0/3,526`. The corresponding `both wrong` counts are retained in the canonical E38 report under each model’s `primary_pairs`.

All four seed1/2 models pass both registered directional criteria: B@4572 has fewer full-trace errors than A, and B8000 has fewer full-trace errors than A. The accepted seed0 E36 endpoints also pass both comparisons: float A 4,006 → B@4572 14 and B8000 12 full-trace errors; W4 A 4,785 → B@4572 30 and B8000 14. Thus the per-model conjunction is true for seeds 0, 1, and 2 at both precisions. No averaging or p-value claim is used.

## Independent final review

The independent saved raw audit and strict replay returned **ACCEPT**. The raw audit checked saved outputs covering 3,296 scientific forwards, 786,176 cases, 4,673,536 readout positions, and 37,388,288 native internal steps in 10.417973 seconds. It performed zero model forwards, checkpoint loads, or updates; those coverage counts are not additional reviewer compute. The strict replay used 22 endpoint loader calls and 388 checkpoint deserializations, reproduced the same 3,296 forwards and 37,388,288 steps in 615.777502 seconds, and performed zero updates. The saved review artifacts are [raw.json](../runs/e37_e38_review_v3/raw.json) (SHA256 `192545d57a3c000b2098d2b4e396c3b950d1a0286e9d3f56a5a4aa2bb78c803f`) and [replay.json](../runs/e37_e38_review_v3/replay.json) (SHA256 `eb55206a2e221745336a78aa7dea321ec09c547bee80e5977461d32cc0596698`).

## Sampled training attainment

All four E38 models have zero errors on the saved attainment probes. A uses 2,880 cases: L1 576/576, L2 1,152/1,152, and L3 1,152/1,152 full-trace correct. B@4572 and B8000 each use 6,336 cases: L1 576/576 and L2–L6 1,152/1,152 at every length. Every branch is therefore 100% on its sampled probes, above the registered 99% threshold. These probes are sampled and do not establish exhaustive training-set mastery.

## Scope and stop

The primary comparison is finite and uses the common semantic-heldout pool after excluding the union of trained prefix functions. It contains 24 program strings and 23 semantic functions; no new pool discovery was performed. The historical separate six-program L5 regression pool is omitted as registered in the inherited E36 scope. `train_syntax` is union exposure across ancestor/A/B and is not branch-A-only exposure at L4–L6. A and B use the same inference programs and native8 evaluation schedule; they differ in training length, composition diversity, and total training compute. B@4572 matches A in internal training instructions, while B8000 matches A in added updates. The observations do not establish arbitrary-depth or arbitrary-chain execution, an architectural limit, or statistical precision superiority.

The fixed E37/E38 scope is complete. The independent reviewer accepted the saved artifacts, so the registered stop condition is reached.
