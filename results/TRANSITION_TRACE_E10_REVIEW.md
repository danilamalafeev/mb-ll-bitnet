# E10 transition-trace independent review — 6 September 2026

Scope was limited to the E10 research-log entry, `scripts/transition_trace_e10.py`, `tests/test_transition_trace_e10.py`, `results/TRANSITION_TRACE_E10.md`, and `runs/transition_trace_e10/{source_manifest,report}.json`. No E10 code, run, E09 artifact, shared coordination document, or E11 artifact was modified. No full suite, sweep, or new inference run was started.

## Verdict

No material blocker found. The saved trace is internally consistent with fixed `hops=2`, the E09 source manifest, the three E07 auxiliary checkpoints, and the stated inference-only recurrence. The E10 dataset reuse is explicitly exploratory and the report correctly limits first-hit observations to transient diagnostics; neither is a validity defect for this narrowly scoped trace.

## Checks and evidence

- `source_manifest.json` records the E09 manifest fingerprint `c0bbe42ce804e6669d6f1dd3eaf69276c664d48f70000867edb0d54d9c3c8c28`, `exploratory_reuse=true`, and `fixed_hops=2`. The report repeats the same fingerprint and fixed-hop setting. The evaluator constructs every input as `Example(..., hops=2, ...)` from the 512 E09 records, so the trace does not mix hop strata.
- The trace loop is algebraically the same as the model recurrence: structured input encoding and fixed KV cache, shared `original_query`, reader update, residual `after_reader`, `select_block(step)`, output head readout, then state replacement. It records logits after every step 1–32 and never uses a target or predicted class to update state. Targets are computed only after inference with `solve`; graph distances are decoded from the known manifest cycle.
- The saved report has `n=512` and 32 predictions per example for every seed. Current checkpoint SHA-256 values match the report: seed0 `63e3f42ed4c4ea5b4accc45bee8d51d45cfe8dc406f8ac975ac3f58d62bdc9b5`, seed1 `d3ff12f0fb554f564924ebc3186e86284a7b348f37d710e7ea68207944089c94`, seed2 `7029bcc8c61e17b3872cbc5bf05e385bb95d77d8118854dab4c5cb39cc344b7d`. Each recorded objective is `aux_intermediate_weight1`; the run reports `no_training=true`.
- E09 reproduction is exact for every seed at budgets 4, 8, 12, and 16 (`e09_h2_prediction_match` is `true` in all 12 checks). Ordinary-forward prefix checks are `true` at 4, 8, 12, 16, and 32 for all seeds.
- Independent artifact recount used the E09 manifest records and saved E10 predictions. For each seed, the decoded graph-distance matrix exactly matches `decoded_graph_distance` in `report.json`; all distances are in 0..15. Every independent `first_hit_distribution` exactly matches the saved histogram, and each histogram totals 512 for every target k. The independent `counts_at_4k` also exactly matches the report:

  | seed | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 0 | 512 | 512 | 192 | 12 | 4 | 7 | 13 | 9 |
  | 1 | 512 | 512 | 374 | 70 | 31 | 22 | 20 | 18 |
  | 2 | 512 | 512 | 512 | 512 | 506 | 499 | 492 | 481 |

- As a compact first-hit audit, the independent `never` counts for target k=1..8 are respectively: seed0 `[0,0,17,182,419,471,479,488]`, seed1 `[0,0,9,159,326,415,446,459]`, seed2 `[0,0,0,0,3,6,14,25]`. These match the stored report. The first-hit numbers can include transient guesses and do not establish a stable step-by-step algorithm, as the E10 report states.
- Targeted semantic tests were run with:

  `PYTHONPATH=. .venv/bin/pytest -q tests/test_transition_trace_e10.py` → **2 passed**.

  The tests cover inverse cycle distances and first-occurrence/`never` semantics. The artifact recount supplies the missing run-level semantic check.

## Limitations and test gap

The E10 test file does not execute a model trace or independently assert every saved count; it tests only `distance_map` and `first_hit` on small hand-built examples. The report's saved prefix checks, exact E09 prediction reproduction, and independent recount verify the current artifact, but a future implementation regression could evade the two unit tests. This is a regression-coverage gap, not a blocker for the completed run.

E10 deliberately reuses the E09 dataset, so it is not an independent generalization evaluation. First-hit histograms describe when a target class is first guessed and may reflect transient logits. The norm and relative-update summaries are descriptive aggregates; they do not identify a causal mechanism or establish that growing state norms cause the degradation. The observed counts therefore support a trace diagnostic, not a proof of an algorithmic transition schedule or its cause.

