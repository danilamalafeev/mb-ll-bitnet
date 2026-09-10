# E16 independent final code review

Status: **CLEAR** for exactly the registered seed0 QAT/float pair, fixed u2000,
followed by seen32 x train192/validation32 evaluation and independent result
review. Reviewer: Astra/low, 2026-09-07. No pilot training was performed by this
review. Earlier HOLD remains preserved in E16_FLOAT_QAT_CODE_REVIEW.md.

## Frozen reviewed files

| File | SHA256 |
|---|---|
| looped_bitnet/float_qat_e16.py | 52319d0ffcee041cbf3ac156fb39a3959460dab5fb679e6190eba416f92de7f3 |
| scripts/float_qat_e16.py | eaf01f8a50942a85f008393f93651298f776985a33612f8dcf4d20396868977f |
| tests/test_float_qat_e16.py | acc587eb571f0e257c0408d432cb0f12d3d1eee861f102879f5877dbb62a5928 |
| results/E16_FLOAT_QAT_PROTOCOL.md | 6d02ae3285681595241cbed0e2a2375c5dc2cba28cf3db6dabe6b818d6f34887 |

## Evidence and closure

Independent focused test rerun: **8 passed in 6.25 s** using the workspace
.venv Python. This includes exact torch.equal QAT logits/loss/gradients and
post-update parity, identical independent initial masters and 152768 parameters,
all14 replacements, exact stream/target digest, 6144/6144 exposure, both-arm
checkpoint state/logit round trips and wrong-arm/prefix/update/manifest/optimizer
rejection. Additional reviewer controls rejected a wrong source map and the
reserved test split before inference, verified disjoint default preflight/run
paths and rechecked **all101 protected hashes**. Executor reported90 full-suite
tests passed; the reviewer did not duplicate that broader run.

The six bounded HOLD findings are resolved:

- Default preflight is the separate runs/e16_float_qat_preflight directory.
- Recovery discovers the two exact final checkpoint paths and validates both
  before inference; no report/arm_complete prerequisite and no training call.
- Evaluation sets eval mode; normal final evaluation and fresh recovery set
  deterministic CPU4 execution. Frozen E15 evaluator uses inference_mode.
- Exact parity/logit controls now use torch.equal; strict loader checks format,
  optimizer, source/config/protocol/manifest provenance, full and final-prefix
  digest, arm/inventory, fixed update2000 and model-state digest.
- Exposure records canonical frequency digests in frozen program/state order.
- Target serialization is checked against a fixed expected digest and against
  the preflight manifest before training/recovery evaluation.

The comparison preserves the existing model methods and all non-BitLinear
components; it jointly removes ternary-weight and int8-activation fake
quantization. Training consumes the same frozen E15 stream and loss with fresh
independent optimizers, fixed4 substeps and no validation selection. Both final
checkpoints are saved before normal final evaluation. Cost arithmetic matches
4000 total updates,256000 examples and2048000 internal state updates.

Per-program final/register/prefix/full-trace counts and per-case final outcomes
are retained across both splits. Length means and equal length2/length3
composition means are correct. Paired counts use aligned case booleans;
float-minus-QAT and train-minus-validation differences remain distinct.
No additional prefix macro is required: all prefix counts are retained.

## Limits and next action

This clears implementation, not any scientific result. Run one fixed pilot;
do not extend seeds, modify inputs/steps, open forbidden/test/secondary scopes,
or change hashes. Stop on invariant/nonfinite failure; incomplete training is
not automatically restarted. Checkpoint-only recovery is inference-only and
returns JSON, which can be saved as a new recovery artifact without overwriting
the original evidence. Final result review must independently recount both
checkpoints, paired marginals, budgets, exclusions and protected hashes.
