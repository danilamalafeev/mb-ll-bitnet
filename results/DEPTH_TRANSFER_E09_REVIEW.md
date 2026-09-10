# E09 depth-transfer independent review — 6 September 2026

Scope was limited to `scripts/depth_transfer_eval.py`, `tests/test_depth_transfer.py`, `results/DEPTH_TRANSFER_E09.md`, `runs/depth_transfer_e09/{manifest,report,verification}.json`, the E07 auxiliary checkpoint metadata/state, and the model/data definitions needed to interpret them. No code, checkpoint, run, shared log, or E09 manifest/report was modified.

## Verdict

No material blocker found for the reported E09 run. The saved predictions, manifest, checkpoint hashes, source-level evaluator constraints, and the paired test construction are mutually consistent. The only test-coverage gap is that the semantic-count test currently checks shapes and lengths; the artifact-level recount below supplies the missing independent semantic check for this run.

## Checks and evidence

- `manifest.json` contains 512 records, 512 unique cycle tables, valid 16-object permutations, valid row orders, and the same transition table, start, and row order used to construct each hop stratum. Recomputing the manifest fingerprint after removing its stored `fingerprint` field gives `c0bbe42ce804e6669d6f1dd3eaf69276c664d48f70000867edb0d54d9c3c8c28`, exactly matching the manifest and report.
- Recomputing `split_for_table` for every manifest table gives `train=0`, `validation=0`, `test=512`.
- The E09 manifest was written before the evaluator enters its model-evaluation loop (`scripts/depth_transfer_eval.py`: manifest write before `evaluate`), and the report points to that manifest fingerprint.
- Each checkpoint was loaded with `torch.load(..., weights_only=True)`, checked for `objective=aux_intermediate_weight1`, diagnostic status, readout step 4, final step 8, `resume_supported=false`, expected seed, quantization, and structured reader, then loaded into `ReasoningModel` with the default strict state-dict check and put in eval mode. Current SHA-256 values still match the report: seed0 `63e3f42ed4c4ea5b4accc45bee8d51d45cfe8dc406f8ac975ac3f58d62bdc9b5`, seed1 `d3ff12f0fb554f564924ebc3186e86284a7b348f37d710e7ea68207944089c94`, seed2 `7029bcc8c61e17b3872cbc5bf05e385bb95d77d8118854dab4c5cb39cc344b7d`.
- The longest encoded hops4 input is 55 tokens and the checkpoint model limit is 64. The evaluator explicitly checks this before evaluation; no limit change is made.
- `data.encode` includes only the edge table, start object, STEP tokens, and END. The evaluator passes only `input_ids` to the model. Targets and trajectory nodes are computed separately with `solve` for counting; no target or predicted class is fed into the recurrent state. `forward_with_readouts` uses the shared output head and does not feed readout logits back into the state.
- Prefix equivalence is checked in the evaluator for budgets 4, 8, 12, and 16 by comparing direct `model(..., steps=budget)` logits with the corresponding readout. The address-specific test run was:

  `PYTHONPATH=. .venv/bin/pytest -q tests/test_depth_transfer.py` → **4 passed**.

- An independent recount from each saved prediction list and the manifest records reproduced every saved `final_target_correct`, `trajectory_target_correct`, `n=512`, and prediction-list length. The full matrix is:

  - seed0: h1 `512/512, 0/512, 0/195, 0/12`; h2 `0/512, 512/512, 320/192, 32/12`; h3 `0/512, 0/512, 196/196, 456/12`; h4 `0/512, 0/512, 0/200, 12/12`.
  - seed1: h1 `512/512, 0/512, 0/371, 4/71`; h2 `0/512, 512/512, 137/374, 10/70`; h3 `0/512, 0/512, 378/378, 401/72`; h4 `0/512, 0/512, 0/380, 73/73`.
  - seed2: h1 `512/512, 0/512, 0/512, 0/512`; h2 `0/512, 512/512, 0/512, 0/512`; h3 `0/512, 0/512, 512/512, 0/512`; h4 `0/512, 0/512, 0/512, 512/512`.

  In each cell the pair is `final-target / trajectory-target`; the trajectory target is `f^(budget/4)(start)`. Therefore the requested main diagonal is independently confirmed: seed0 `[512,512,196,12]`, seed1 `[512,512,378,73]`, seed2 `[512,512,512,512]`.

- Novelty was checked against all 37 saved historical `eval_sets.json` files (3,200 unique old tables), regenerated E07 validation/in-distribution/longer suites (1,280 unique tables, using the checkpoint config), and both sides of all 512 E08 pairs (1,024 unique tables). E09 has zero table overlap with each group and zero combined overlap. All 512 E09 tables are therefore fresh relative to the checked E07 suites and E08 pair tables. The E09 set is one shared 512-table dataset across all seeds and hop strata; it is not three independent 512-table datasets or 1,536 new tables.

## Limitations and test gap

`test_evaluate_reports_independent_final_and_trajectory_targets` currently asserts only result keys, denominators, and prediction-list shapes. It would not catch a wrong target computation or incorrect semantic counts by itself. The independent artifact recount above catches those errors for the saved run, so this is a future regression-test gap rather than a blocker for E09.

The manifest retains the E07 data configuration field `test_max_hops=3`, while E09 manually constructs the additional hops4 examples. This does not invalidate the run: the evaluator explicitly constructs hops4, computes length 55, and checks it against `max_tokens=64`; it is a metadata clarity issue for future reruns.

The result remains inference-only evidence for this particular fresh cycle16 table distribution and the existing E07 auxiliary objective. It does not establish transfer to other table distributions, object counts, or independently sampled datasets per seed.

