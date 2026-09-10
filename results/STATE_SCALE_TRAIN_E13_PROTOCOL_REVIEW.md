# E13 protocol review: trained cycle-scale normalization

## Verdict

The proposed E13 comparison is conceptually testable and keeps the important E07 controls: the same auxiliary objective, four-block cycle, two trained cycles, parameter-free intervention, and paired initialization/data. One material protocol gap must be closed before training: the inference condition is ambiguous. Since E12 already showed that post-hoc normalization can change continuation, “apply the same rule at inference” must say whether the baseline checkpoint also receives the rule. Otherwise the result mixes a training effect with a test-time intervention effect.

The minimal resolution is a 2×2 cross-evaluation on the same test manifest:

| checkpoint trained with | inference without reset | inference with reset |
|---|---|---|
| baseline | B0 | B1 |
| normalized | N0 | N1 |

The natural deployment comparison may be N1 versus B0, but N1 versus B1 is the training effect under a common inference rule, while B1 versus B0 and N1 versus N0 expose the test-time effect. Report all four; do not select the preferred pair after seeing test results.

## Protocol conditions to freeze

- Require `num_blocks=4`, `model.steps=8`, `d_model=64`, fixed E07 structured-QAT architecture, batch 64, 2,000 updates, the same optimizer hyperparameters, and the same `CE(readout4, f(start)) + CE(readout8, f²(start))` objective with weight 1.0. Confirm equal trainable parameter counts and record them.
- Build one initial model state per seed, record its digest, and copy that exact state into the baseline and normalized arms. Reset the training stream to the same state for both arms, or precompute a batch-sequence digest. Use separate optimizers with identical hyperparameters. Record seed, stream seed, initialization digest, examples seen, and selected update for each arm.
- The normalized trace must be ordered as follows: recurrent step; if a readout is requested, compute and save its logits; then, only for the normalized arm, apply

  `m + (h - m) / max(||h - m||_2, 1e-8)`.

  At training step 4, the unnormalized readout supplies `f(start)` loss and the reset affects steps 5–8. At step 8, compute the final `f²(start)` readout and its loss before the step-8 reset; that reset is unused by the eight-step training objective. At inference steps 4, 8, ..., 32, always read out before resetting; the reset after the final step is irrelevant and may be skipped. A unit test must catch accidentally applying the step-8 reset before the final loss.
- With dimension 64 and nonzero centered state, radius 1 implies centered RMS `1/sqrt(64)=0.125`. State the radius and epsilon as fixed protocol constants, with no coefficient sweep. The rule preserves the per-example mean and centered direction; the zero-centered case needs an explicit finite-output test. Confirm the implementation uses autograd through the norm, with no `detach`, and test finite/nonzero gradients on a nondegenerate toy state.
- Generate the common 512-table cycle16 test manifest with stream seed `20260910`, fixed input hops 2, and a fingerprint written before test evaluation. Check zero overlap, fail-closed, against all persisted historical `eval_sets` tables, reconstructed E07 suites, E08 pair sides, E09, E10, E11, and E12 manifests. The corrected E12 historical reader uses `eval_sets["sets"]`; E13 must use the same schema explicitly. Record scope cardinalities and overlaps, including E12.
- Keep test tables out of all gate and checkpoint decisions. Select checkpoints with the same predeclared validation rule for both arms (the E07 rule is final validation accuracy, then final validation loss), then apply the extension gate at the selected checkpoint: each arm must have both final and intermediate validation accuracy at least 95%. Record the gate for every seed and do not retry, retune, or change the budget after a failed gate.
- The test is fixed-hop continuation on a new table set, not a new depth or hop distribution. Report all `k=1..8` counts at readouts 4, 8, ..., 32, paired k8 wins/losses/ties and delta for every evaluated seed, and identify the common 512 examples used across seeds. Do not call a result “across seeds” if the validation gate excludes seeds 1 or 2.

## Minimal tests and controls

The E13 test file should include these semantic checks:

- A hand-built full 8-step reference verifies four-block ordering, reset-after-readout ordering, the exact auxiliary readouts, and that the reset after step 8 cannot affect the final loss.
- The normalization helper preserves means and centered directions, reaches centered RMS 0.125 within tolerance for nonzero states, remains finite for zero-centered states, and propagates gradients without detach.
- A paired-training fixture verifies identical initial digests, identical first batch tensors/targets, equal parameters, and distinct optimizer objects. The saved run should also record arm data-stream fingerprints.
- A four-arm inference fixture verifies B0/B1/N0/N1 schedule semantics and exact no-op behavior where applicable. Save per-example predictions and enough metadata to reproduce the cross-evaluation.
- Checkpoint metadata must record objective, normalization rule, schedule, radius, epsilon, seed, updates, validation-selection rule, and initialization digest. Verify no test target or predicted class enters the recurrent state; `f(start)` and `f²(start)` are loss labels only.

## Primary endpoint and stop rule

The protocol currently names the k8 paired comparison but does not predeclare what numerical result counts as support for “improves depth continuation across seeds.” Add that decision before training. A minimal inherited practical threshold would be `normalized - baseline >= 26/512` on k8 for each completed seed under the declared primary pair, with wins/losses shown separately; if that threshold is not adopted, report the result as descriptive and avoid a support/failure label based on an unplanned margin.

Run seed 0 as the paired pilot. The validation gate controls whether seeds 1 and 2 are trained; it never uses test results. If a later seed fails the gate, stop that seed without retries and state that the cross-seed claim is unresolved. After the planned seeds and the four-arm test evaluation, stop: no coefficient/schedule sweep, extra seeds, test-based checkpoint choice, or follow-up training in E13.

The E12 measurements informed the normalization mechanism, while radius 1 is a new fixed geometric choice rather than a fitted coefficient. The final interpretation should therefore separate: training compatibility with a fixed centered-radius rule, the independent inference effect, and any evidence of better long-horizon continuation. Even a positive N1 result does not establish that scale is the sole cause of E09 degradation or that the rule is a learned architectural component.

## Protocol addendum

The registration resolved the inference ambiguity before training: the primary comparison is normalized-training with normalized inference (`N1`) versus baseline-training with no inference reset (`B0`), with the full 2×2 cross-evaluation retained. The preregistered practical criterion is `N1 - B0 >= 26/512` at k8 for seeds 0 and 1 and nonnegative for seed 2. A stronger result is recorded separately only if normalized training reaches at least `487/512` at every k3–k8 readout for every evaluated seed. Seed 0 alone gates whether seeds 1 and 2 are started; once extension is authorized, both use the fixed budget regardless of later validation outcomes, avoiding convenient-seed censoring. The original invalid seed-0 pilot is frozen and excluded from E13 conclusions; this review concerns the corrected run protocol.
