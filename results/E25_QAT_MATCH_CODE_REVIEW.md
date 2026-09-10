# E25 independent code review — CLEAR

Independent Astra/low review, 2026-09-07. CLEAR authorizes only the registered three-seed QAT experiment at the exact reviewed hashes below. No E25 scientific forward or training update occurred in this review. Final results require separate independent replay and provenance/count review.

## Exact reviewed artifacts

| Artifact | SHA256 |
|---|---|
| looped_bitnet/qat_match_e25.py | 28046d2ff5ad69f61e1c02f3a15260fc4f85b1f1c39d7a43d1081523670ca2d2 |
| scripts/qat_match_e25.py | a06281c5df01dd9cd73def44d4e5026e95ea66ba9f2dfb0c2493e7630df26bb5 |
| tests/test_qat_match_e25.py | 8a5f70874aa59ad1d18e7c9a241a3fe8a41a177b434b0918fd75912f4e413468 |
| results/E25_QAT_MATCH_PROTOCOL.md | 8e8e0718c2d88578dabcc1834a44999f11076ce8f9f0467afc0731017576055a |
| results/E25_INITIAL_REFERENCE.json | 0a1f3538841c631a4e2f6fe17fcc9334dda4af40f6974b7eb773c5f4d3a18a2d |

All 249 protected hashes match after review. Source/config/data/initial/float-final reference identities are frozen transitively by the manifest and checkpoint manifest digest.

## Design and pairing

The estimand is retention of the registered task threshold by the existing ternary-weight plus INT8-activation fake-quantization package at matched training budget. It does not isolate weight quantization or claim statistical equivalence, packed storage, acceleration or general reasoning.

The exact fourteen BitLinear module names are enforced, not merely their number. All initial named parameters and state tensors match accepted float masters in key/order/dtype/shape/value using torch.equal and separate storage. Input projections remain Linear, native steps remain eight, and parameter count is 151232. A real BitLinear-to-Linear replacement is rejected. Saved seed0 E18 and seeds1/2 E22 initials and root independent reference are checked.

The pretraining RNG correction was independently confirmed without forwards: actual seed0 E20 training begins with bare manual_seed(0), since its constructors preserve the caller RNG; E22 seeds1/2 explicitly restore their post-constructor RNG. The corrected QAT initialization follows these actual per-seed float training states, rather than using the E22 seed0 constructor RNG. The canonical reference records this distinction before training.

## Implementation checks

- The stream verifies all eight repeats of accepted base2000 batches and targets. Scientific execution checks exact 16000-update stream digests, uses the unchanged accepted update/AdamW/loss/clipping path, restores matched initial RNG after constructors, and saves/evaluates only each final. No float retraining occurs.
- All three seeds continue after a completed false task predicate. Exceptions, nonfinite outputs/gradients or other technical failures suspend remaining seeds and preserve actual completed work and artifacts. Composition reporting is not gated by the seen prerequisite.
- QAT checkpoint reload constructs the QAT model, checks exact inventory, tensor metadata/digests, optimizer fields/config/moments/steps, seed/update/schema/provenance, actual cost metadata and training mode, then restores saved RNG after constructor scopes. It cannot silently substitute float execution.
- Evaluation checks immutable model/optimizer/RNG and restored training mode. Shared accepted predicates independently enforce six primary rows at >=244/256, with the control excluded. QAT-minus-float paired directions, identities/targets/strata, marginal counts, seen deltas and rates are checked.
- Completed top-level forwards and attempted forward workload are distinct counters. A failure inside a forward cannot establish an exact partial internal-step count; this limitation is explicit, without misreporting attempted full workload as completed work. Checkpoint costs must equal actual completed training costs. Existing outputs are refused.

## Independent verification and cost

Command: `.venv/bin/pytest -q tests/test_qat_match_e25.py --basetemp=/private/tmp/e25-independent-review-qa`

**7 passed in 5.82s.** Real legal seen-only QAT QA covers all three seeds, final save/strict reload, next-update equality for model/optimizer/RNG, false-predicate continuation, technical failure after one real update, finite accepted updates, exact counters and overwrite refusal. Metadata tests cover actual QAT inventory replacement, checkpoint tampering, paired state/target/stratum/marginal rejection, win/loss direction, and 244 versus 243/control separation.

This seven-test execution used runner hash aac11d91efbad80d11e14af394c7aa82a7cfc532e84b5978071cb534a208c657. The final narrow patch added nonfinite-output checking in the completed-forward hook and actual-versus-manifest training-cost equality before checkpointing. The final patch was inspected and verified without repeating training: direct hook callback checks accept finite synthetic tensors and reject NaN, infinity and invalid output type; retained real QA records satisfy the new cost equality. These callback invocations were not model forwards and are excluded from model-work accounting.

Additional independent zero-forward checks reject actual model tensor, optimizer moment and RNG tampering; verify the canonical scientific manifest cost of 16000 forwards, 1024000 examples, 2048000 readouts and 16384000 internal steps per seed; and verify all protected hashes.

Reviewer QA work: **10 successful updates, 20 training examples, 160 training internal substeps** (4 main-run updates plus 6 reload-identity updates); **3 evaluation forwards, 6 cases, 6 readouts, 48 evaluation internal substeps**. One injected evaluation exception occurs before its first forward after a counted successful update; remaining seeds do not start. No failed update was retried. Executor QA costs are separate and must be added, not replaced by these reviewer totals.

No remaining blocking finding. Root may freeze canonical preflight and launch one fixed registered scientific run. Any source change invalidates clearance for that changed source until reviewed. The stop condition remains all three fixed finals and independent result review, regardless of numeric outcome.
