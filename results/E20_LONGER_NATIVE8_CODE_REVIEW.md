# E20 independent code review — CLEAR

Reviewer: Astra/low,2026-09-07. Clears only the prospectively registered fresh
native8 seed0 run to fixed8000 updates, with observations2000/4000/8000 and
mandatory exact E18 prefix reproduction before2001. No scientific E20 training
was performed in this review. No source or shared coordination file was edited.

## Reviewed source hashes

- module: `8f738b0c881eaa1f0488f44b46d63591ac77bc276d811b016f44fab8b06e8c12`
- runner: `bd25091100c1b1cc46880bb98c65c128466e07233e6bd4dd47fbe0bac1b3893a`
- tests: `f1d20bc393e709d8ef735c2a713181cf826354be82728d08e0ab6516d593df69`

Freeze these with protocol/dependency/reference hashes in fresh preflight
before update1. Any source change invalidates this hash-specific clearance.

## Evidence and executable-path controls

Independent focused rerun: **3 passed in36.41s** with
`.venv/bin/python -m pytest -q tests/test_longer_native8_e20.py`.
No broader suite was repeated. The tests execute the ACTUAL `_train` loop,
including updates, multiple atomic checkpoints, progress records, evaluation,
training-mode restoration and final report writing. They do not merely perform
manual optimizer steps outside the runner.

The legal length2 tiny test crosses milestones2/4; compared with the same
runner without evaluation, model, AdamW and torch RNG digests remain exact.
Reloaded model/optimizer from2 reaches the exact model and moments at4 after
the next two actual update-helper calls. The intentionally wrong prefix stops
at2 before any update3 or final4 checkpoint. Tampered native budget, stream,
initial/source/optimizer/RNG digests are rejected, as are QA checkpoints loaded
through the scientific mode. Missing-milestone recovery remains explicit.

Additional independent controls executed from `/tmp/e20_review_extra.py`:

- Exercised the actual scientific `comparisons()` helper using frozen legal
  E18 native4 rows repeated at2000/4000/8000. Every expected difference is zero;
  the helper executes successfully with all registered groups and split keys.
- Executed an additional two-update, one-example length2 QA runner fixture,
  then recovered its existing two milestones with the training entry point
  patched to fail if called. Recovery succeeds without a report dependency;
  checkpoints are not rewritten. Private QA milestone/load injection was used,
  never a scientific CLI override. These fixtures were removed afterward.
- A synthetic falsely labeled scientific2000 checkpoint with the wrong prefix
  model digest was rejected explicitly. It was temporary QA, not evidence.
- All176 protected historical file hashes verified unchanged.

Independent checks also verified the E18 reference manifest, checkpoint byte,
report byte and final model digest constants against the accepted artifacts.
The complete repeated stream, all four phase digests, exact initial state,
parameter membership and8192000 planned training substeps pass focused checks.

## Important source-path conclusions

One fresh model and one AdamW instance persist across the complete loop. No
moment reset occurs at milestones. Checkpoints save both states, RNG and
cumulative cost before inference. The prefix gate checks exact keys, dtypes,
shapes and torch.equal values against the fixed E18 reference, then permits
further training. Failed-prefix2000 checkpoints cannot later masquerade as
valid recovery milestones: scientific loading also requires the registered
E18 model digest. Evaluation restores training mode and verifies unchanged
model/optimizer/torch RNG state before the next update.

The loader reconstructs the correct native8 model, validates scientific/QA
mode and every expected metadata field, checks final state/optimizer/RNG
hashes, frozen hyperparameters and complete AdamW states/step counters.
Recovery only reads valid existing milestones and reports missing ones; an
absent8000 milestone is not called a completed final. Stored optimizer state
does not authorize training continuation. Preflight checks protected files,
full source/reference/config/protocol identity and saved initial state; output
roots and checkpoints refuse overwrite. Dependencies include the core model,
configuration, data and engine as well as reused experiment helpers.

Two defects in the initial partial module (base-stream constant typo and wrong
E18 return-value unpacking) and the failed-prefix recovery gap were corrected
before this stable review. They produced no scientific training attempt.
The earlier E18 lesson is addressed by actual runner-path QA here.

## Scope, QA cost and stop

The independent focused rerun used ten tiny actual runner updates plus two
post-load update controls:24 QA examples and384 recurrent substeps. The extra
recovery fixture used2 QA updates/2 examples/32 substeps. This verification
work is separate from the planned8000/512000/8192000 scientific budget and from
the executor's earlier QA work. No failed scientific attempt or retry occurred.

No blocking finding remains for the frozen CLI path. Native8/u8000 versus
native4/u2000 is an unequal-budget catch-up comparison, not a fair architecture
ranking or an inference-only experiment. No outcome selects an extension.
Stop on invariant/nonfinite failure; otherwise stop after fixed8000 and
independent result review. E19 must not alter E20's stream or settings.
