# Independent review: E12 state-scale schedule

## Verdict

The saved E12 inference artifact is internally consistent, and the predeclared schedule criterion is met. No material blocker remains after the executor's narrow historical-scope parser fix. The result is a diagnostic evaluator intervention on one new 512-table set; it is not evidence for a learned scale mechanism or a general causal explanation.

## Independent evidence

- Recomputing the manifest fingerprint over `manifest.json` without its stored `fingerprint` gives `3f615c6324a0044dc3dee485d0e91377c014b087eb1ec8e2b8db8f014124f530`, matching the manifest and `report.json`. The manifest has 512 unique tables; all 512 are valid 16-object Hamiltonian cycles, assigned to the `test` split under `split_seed=1729`, with fixed input hops 2. Every seed and arm uses this same table set.
- A correct independent walk of all 37 historical `eval_sets.json` files (through their `sets` object) found 3,200 unique tables. The other novelty scopes contain 1,280 reconstructed E07 tables, 1,024 E08 pair-side tables, and 512 each from E09, E10, and E11. Intersections with the new manifest are zero for every scope.
- The original saved report records `e07_eval_sets: 0`, but that zero-overlap claim was unvalidated because its pre-fix reader iterated the JSON top level and collected an empty historical scope. The executor's final reader now parses `obj["sets"]` and fails closed on missing or malformed sources; `_old_tables()` currently returns the expected cardinalities above, and the added source regression test passes. The saved run data were not changed.
- All three checkpoint files still match their recorded before/after hashes and strict state-dict loads succeed. Each is tagged `objective=aux_intermediate_weight1`, `resume_supported=false`, with QAT and structured reader enabled:

  | seed | SHA-256 |
  |---:|---|
  | 0 | `63e3f42ed4c4ea5b4accc45bee8d51d45cfe8dc406f8ac975ac3f58d62bdc9b5` |
  | 1 | `d3ff12f0fb554f564924ebc3186e86284a7b348f37d710e7ea68207944089c94` |
  | 2 | `7029bcc8c61e17b3872cbc5bf05e385bb95d77d8118854dab4c5cb39cc344b7d` |

- The inspected trace uses only encoded table/start/STEP input, a fixed K/V cache, current state, reader update, and cyclic FFN block. `solve(...)` is called only while constructing evaluator counts; no target or predicted class enters the recurrence. `r4` is captured from each example's own post-step-4 state, and each reset is applied after that step's readout. The saved controls report all baseline-forward prefix checks true at readouts 4 through 32.
- I replayed the saved manifest in batches for all three checkpoints. Baseline trace logits equal ordinary `model.forward(..., steps=s)` logits exactly at every readout 4, 8, ..., 32, and baseline/noop logits are exactly equal at every readout for all seeds. Saved baseline predictions also match this replay. Single and repeated prediction arrays are exactly equal for all 512 examples through readout 12 for every seed; divergence begins after repeated's second reset at step 12, as scheduled.
- Every saved reset row has 512 finite values. Recomputing `factor=r4/max(rms_before,1e-8)` gives maximum absolute error below `6.0e-8`; recomputed post-reset centered RMS differs from the saved per-example `r4` by at most `3.0e-8`. The four arm `r4` arrays are exactly identical per seed.

## Recomputed counts

I independently walked each manifest transition table with `solve(transitions, start, k)` and compared it with every saved prediction at step `4k`. Each cell below is out of 512 and matches `report.json`.

| seed | arm | k=1..8 counts at steps 4k |
|---:|---|---|
| 0 | baseline | `512, 512, 196, 13, 10, 8, 15, 11` |
| 0 | noop | `512, 512, 196, 13, 10, 8, 15, 11` |
| 0 | single | `512, 512, 459, 322, 121, 41, 21, 18` |
| 0 | repeated | `512, 512, 459, 363, 291, 238, 213, 178` |
| 1 | baseline | `512, 512, 388, 75, 40, 17, 19, 17` |
| 1 | noop | `512, 512, 388, 75, 40, 17, 19, 17` |
| 1 | single | `512, 512, 506, 459, 247, 109, 60, 43` |
| 1 | repeated | `512, 512, 506, 481, 451, 419, 399, 371` |
| 2 | baseline | `512, 512, 512, 511, 505, 504, 500, 488` |
| 2 | noop | `512, 512, 512, 511, 505, 504, 500, 488` |
| 2 | single | `512, 512, 512, 512, 508, 507, 504, 496` |
| 2 | repeated | `512, 512, 512, 512, 512, 511, 511, 509` |

Direct paired walks of repeated versus single k8 give:

| seed | wins / losses / ties | repeated − single |
|---:|---:|---:|
| 0 | `170 / 10 / 332` | `+160` |
| 1 | `331 / 3 / 178` | `+328` |
| 2 | `13 / 0 / 499` | `+13` |

The predeclared diagnostic threshold of at least `+26/512` for both seeds 0 and 1 is therefore satisfied. Seed 2 improves by 13.

## Tests and limitations

After the final parser and regression hardening, `PYTHONPATH=. .venv/bin/pytest -q tests/test_state_scale_schedule_e12.py` returned **5 passed**. This includes the independent toy full-trace schedule regression, exact toy noop, helper/count checks, and nonempty novelty-source check. The executor's **43-test full-suite** result predates the final parser-only hardening and is not reasserted here.

The saved report was produced before the exact-logit and fail-closed-scope additions, so its `noop_checks` contains only `logits_allclose` and `argmax_equal`, and its persisted E07 old-eval zero-overlap claim was unvalidated as described above. The bounded replay and corrected scope audit close both evidence gaps without changing the run. The targeted toy trace test does not execute all 32 steps of every real checkpoint; the saved prefix controls and exact replay cover the current artifact.

The schedule is one fixed coefficient and one fixed repeated schedule, applied by the evaluator to existing checkpoints on a common 512-table set. It supports the observed schedule comparison under these conditions, while leaving coefficient selection, retraining, independent per-seed samples, and generalization unresolved.
