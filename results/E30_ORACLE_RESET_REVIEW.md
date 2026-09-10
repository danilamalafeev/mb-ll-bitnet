# E30 independent final review — ACCEPT

2026-09-08. Accepted the fixed oracle diagnostic, not a learned L5 correction or length-generalization result.

Independent saved-checkpoint replay at `runs/e30_review_final/` completed exactly 12 real forwards, 3072 unique primitive/state cases and readouts, 24576 internal state steps, zero updates. Attempted and completed counters match. Elapsed 5.374 s (user 3.74 s, system 0.74 s). Full report and manifest exactly equal canonical E30. All six checkpoint evaluations preserve model, optimizer, torch/Python RNG and module modes. All 321 protected hashes and two frozen source hashes remain unchanged; E30 reference hashes match.

Separate stdlib recount (`results/E30_review_raw.py`, evidence `results/E30_review_raw_counts.json`) recomputed the DSL, 256-state prefix bijections, opcode/state lookup keys, all 9216 lifted identities/true-prefix/targets/saved predictions/initial strata, prefix-four correctness, paired counts and marginals for each model/seed and aggregate, all three prefix subsets and all original-state strata, descriptive per-program counts/244 flags, both primary diagnostic predicates and accounting. It does not use the production paired scorer.

All 12 fresh primitive cells are 256/256 correct. `fresh_exact_all=true`; `late_error_rescue_all=true`. All 9216 lifted outcomes are fresh-correct: 359 recovered, zero introduced. The prefix-four-correct subset contains 9133 cases, including 283 saved final errors; all 283 are recovered with zero introduced errors. The remaining 83 cases recover 76 saved final errors and retain seven already-correct finals. Empty saved-error denominators remain null.

The oracle supplies true numeric state and starts a fresh input/cache/latent context. This supports availability of the local mapping despite failures in the original long trajectory. It does not isolate hidden state, cache, reader or position/history as the cause, does not supply a free-running repair mechanism, and does not change E28 gates. Lifted comparisons reuse 3072 actual primitive evaluations and are not 9216 independent forwards. Fixed scope complete; no additional experiment warranted by this review.
