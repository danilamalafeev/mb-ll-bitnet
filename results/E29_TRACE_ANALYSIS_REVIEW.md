# E29 independent review — ACCEPT

2026-09-08. This gate accepts the descriptive analysis of saved E28 L5 traces only. It does not accept a new trained result or change E28 decisions. No training, model forward, torch import, or checkpoint load was performed.

The frozen protocol covers the requested scope and explicitly limits interpretation. Reviewed `scripts/trace_analysis_e29.py` and its six synthetic boundary tests. Independent stdlib recomputation in `results/E29_review_independent.py` reconstructs ADD modulo 16, XOR, and SWAP targets from raw saved initial states and programs, checks saved prefix/final/full-trace/first-divergence/recovery counters, and verifies 9,216 distinct cell-case traces (46,080 readouts).

Independent recomputation agrees with production on all six cells' final/trace error counts, every opcode-position at-risk numerator/denominator, predefined feature bins and rates, first-error components, and SWAP patterns split by equal/unequal pre-operation inputs. It also agrees on all 15 pairwise final and trace overlaps, family and all-cell intersections/unions, seed-1 contrast counts, and deterministic top-10 case ordering/membership counts. Pooled tables equal sums of cell tables. Production explicitly revalidates selected program/state identity, strata and saved per-stratum/same-seed counters.

Final errors for W4 seeds 0/1/2 are 68/21/62; trace errors 68/22/64. Float final errors are 138/6/64; trace errors 141/7/64. The differences correspond to 0/1/2 and 3/1/0 cases respectively that finish correctly after a prior decoded error.

Verification:
- `python3 results/E29_review_independent.py`: PASS; raw counts saved in `results/E29_review_raw_counts.json`.
- `python3 -m unittest tests.test_trace_analysis_e29 -v`: all six tests PASS (recovery/risk, semantics, keyed alignment/duplicates, malformed identity, empty overlap, equal-input SWAP).
- Fresh reviewer rerun `results/E29_review_exact_rerun.json` is byte-identical to `results/E29_TRACE_ANALYSIS.json`.
- Frozen E29 reference hashes pass independently; production validates all inherited protected/source hashes before and after analysis. It only reads checkpoint bytes for hashing.

No blocking findings. First divergence is an observed decoded-output boundary, not evidence that hidden computation was correct beforehand. Five of six selected programs end in SWAP, so raw SWAP error mass cannot identify an opcode mechanism. At-risk feature comparisons are descriptive and correlated across programs/cells; arithmetic carry descriptors do not establish a carry mechanism. Equal-input SWAP is correctly excluded from wrong-identity classification. Seed-1 improvement is a fixed three-seed observation; general seed or causal claims are unsupported. Length and program composition remain confounded.
