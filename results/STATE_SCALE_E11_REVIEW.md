# Independent review: E11 state-scale intervention

## Verdict

**No material blocker found.** The saved E11 artifact is internally consistent with the stated inference-only protocol. The manifest is fresh and fixed, checkpoints are the expected E07 auxiliary structured-QAT checkpoints, recurrence controls pass, targeted E11 tests pass, and an independent recount from the saved predictions reproduces every reported count and every expected count.

The result supports the bounded diagnostic claim that this one centered-RMS intervention changed continuation behavior on this particular fresh 512-table set. It does not establish a trained architectural component or a general causal explanation.

## Evidence

### Manifest and split

- Recomputing the canonical SHA-256 over `manifest.json` without its stored `fingerprint` gives exactly `98ce79f0db6d81bb379fc80d419cd3f59f72758a76c76f30c286d28b3b99c68a`, matching both the manifest and `report.json`.
- The manifest contains 512 records and 512 unique transition tables. All 512 are valid 16-object Hamiltonian cycles, all are assigned to the `test` split, and all starts and row orders are valid. `fixed_hops=2`; the evaluator uses 32 recurrent steps and readouts at steps 4, 8, ..., 32.
- Independent overlap checks found zero overlap with 37 saved historical `eval_sets.json` files (3,200 unique tables), regenerated E07 validation/ID/longer suites (1,280 unique tables), the E09 manifest (512 tables), and both sides of all E08 pairs (1,024 tables). The E11 tables are one common 512-table set shared by all three seeds.
- In `scripts/state_scale_e11.py`, the manifest is written at line 180 before the model-evaluation loop begins at line 181. Novelty checks also occur before the write and before evaluation.

### Checkpoints

All three checkpoint files currently have the SHA-256 recorded in `report.json` and load with `strict=True`:

| Seed | SHA-256 | Objective | Resume | QAT | Structured reader | Strict load |
|---:|---|---|---|---|---|---|
| 0 | `63e3f42ed4c4ea5b4accc45bee8d51d45cfe8dc406f8ac975ac3f58d62bdc9b5` | `aux_intermediate_weight1` | false | true | true | pass |
| 1 | `d3ff12f0fb554f564924ebc3186e86284a7b348f37d710e7ea68207944089c94` | `aux_intermediate_weight1` | false | true | true | pass |
| 2 | `7029bcc8c61e17b3872cbc5bf05e385bb95d77d8118854dab4c5cb39cc344b7d` | `aux_intermediate_weight1` | false | true | true | pass |

The checkpoint configurations carry training seeds 0, 1, and 2. The evaluator only reads checkpoints; it contains no checkpoint-save path. Current hashes still match the saved report after review, so no checkpoint mutation is detected.

### Recurrence and feedback controls

- Each model is put in `eval()` mode before loading the state dict. The evaluation loop is enclosed in `torch.inference_mode()`.
- The evaluator passes only `batch["input_ids"]` to the model. Targets are present only in the collated evaluator batch; `solve(...)` is used for evaluator-side count construction. Neither targets nor solved nodes enter `run_loop` or the model recurrence.
- The model recurrence reads a fixed K/V cache, adds the reader update to the current state, and applies the cyclic FFN. Readout logits are not fed back; no target feedback or predicted-class feedback exists in the inspected path.
- Saved `prefix_checks` are true for steps 4, 8, 12, 16, 20, 24, 28, and 32 for every seed, confirming ordinary `forward` matches the trace prefix at all requested points.
- The intervention is applied after the readout at steps 8, 12, 16, 20, 24, and 28. The code captures `r4` immediately after step 4 and uses `mean + (state - mean) * r4 / current_centered_rms` with `eps=1e-8`.
- The formula preserves the per-example mean and centered direction. The zero-centered-state targeted test confirms finite output and identity behavior when the centered state is zero. The no-op branch is structurally an identity: saved baseline/no-op predictions are exactly equal for all seeds and readouts, while the recorded no-op logits are allclose and argmax-equal.
- Each intervention step has 512 saved factors and 512 pre/post logit comparisons. The saved `max_abs_logit_diff_max` values equal the maxima recomputed from the per-example arrays, and `argmax_equal_count` equals the recomputed count. Seed 0 has 511/512 unchanged argmaxes at step 28; all other intervention points and seeds have 512/512. This is expected metadata, not a failure: the intervention is allowed to change the readout class.

Targeted tests: `PYTHONPATH=. .venv/bin/pytest -q tests/test_state_scale_e11.py` -> **3 passed**.

## Recomputed counts

I independently constructed `Example` objects from the saved manifest and recomputed each target as `solve(transitions, start, k)` from the saved predictions. Every denominator is 512; every recomputed value matches both `report.json` and the expected values.

| Seed | Arm | k=1..8 counts at steps 4k |
|---:|---|---|
| 0 | baseline | `512, 512, 195, 15, 12, 13, 12, 7` |
| 0 | noop | `512, 512, 195, 15, 12, 13, 12, 7` |
| 0 | intervention | `512, 512, 468, 384, 308, 250, 220, 189` |
| 1 | baseline | `512, 512, 392, 79, 28, 17, 12, 21` |
| 1 | noop | `512, 512, 392, 79, 28, 17, 12, 21` |
| 1 | intervention | `512, 512, 507, 487, 453, 430, 395, 365` |
| 2 | baseline | `512, 512, 512, 510, 506, 501, 499, 492` |
| 2 | noop | `512, 512, 512, 510, 506, 501, 499, 492` |
| 2 | intervention | `512, 512, 512, 512, 510, 510, 510, 509` |

## Blockers

None for the correctness of the saved E11 inference-only run.

## Limitations

- This is one fixed intervention; there is no coefficient sweep.
- The intervention is applied by the evaluator. It is not a trained architectural component.
- The improvement does not prove that scale is the only, or the naturally correct, cause of the E09 errors.
- The same common set of 512 tables is used for all seeds; this is not 512 independent tables for each seed.
- E11 is a diagnostic causal intervention, not evidence of generalization to arbitrary data or arbitrary depth.
- The run is CPU-only and uses the existing `aux_intermediate_weight1` structured-QAT checkpoints on the cycle16 distribution. It does not test retraining under the intervention.
- The report stores intervention factors and pre/post logit metadata, but not the per-example `r4` and pre-intervention centered RMS arrays. Therefore exact per-example post-intervention RMS equality cannot be numerically recomputed from `report.json` alone; the formula, epsilon handling, and identity cases are verified from source and targeted tests.
- The targeted tests cover the helper semantics and count helper, not a complete replay of the 32-step model trace. The saved prefix checks, source inspection, checkpoint checks, and independent artifact recount cover the current run.

## Recommendation

Accept E11 as a valid, completed diagnostic run with the bounded interpretation above. Preserve the artifacts and do not describe the intervention as a learned architecture or as proof of generalization. No retraining, sweep, or new variant is required for this review.
