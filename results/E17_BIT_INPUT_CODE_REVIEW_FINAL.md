# E17 independent final code review — CLEAR

Reviewer: Astra/low, 2026-09-07. This clears exactly the registered paired seed0
pilot after root freezes preflight provenance. No registered pilot training was
performed by this review. Initial HOLD is preserved in
`results/E17_BIT_INPUT_CODE_REVIEW.md`.

## Repair verification

All five initial finding groups are resolved by source changes plus explicit
independent controls. The loader verifies reconstructed initial/common state
before loading trained weights, then verifies the final model digest. Both
trained encoder modes reload. Environment metadata is present. Initial artifact
validation rejects a self-consistent replacement table and altered digest.
Manifest validation checks the frozen full program/state order, complete source
and dependency maps, protocol/config hashes, seed, update and schedule before
training. The frozen dependency map includes the actual core model and other
imported sources. Paired field semantics and unequal-marginal toy counts are
correct. Recovery requires both checkpoints, never calls training, and works
without a report; missing-arm recovery is rejected.

Independent targeted rerun: **8 passed in10.11s**, using
`.venv/bin/python -m pytest -q tests/test_bit_input_e17.py`.
The executor reports98 full-suite tests passed; this review did not repeat
that broader suite.

## Additional independent controls and exact coverage

The committed tests use batch index666, which has length1 under the frozen
(1,2,3) schedule; their update-parity control is not a multi-prefix control.
Their explicit projection matmul comparison also covers three values, although
the signed-code check covers all16. These coverage limits were found in review,
not hidden behind the passing suite. The following direct controls closed the
registered pre-training requirements without changing frozen source:

- Both actual x/y encoders on all16 values equal an independently constructed
  LSB-first signed-code matrix times the corresponding weight transpose,
  bitwise. Projection hashes also match the committed reference assertions.
- A legal four-example batch from fixed_stream()[1] was asserted length2.
  Original E16 float versus E17 learned had identical initial state, loss,
  every gradient and every parameter after the identical clipped AdamW update.
  Bits received finite nonzero gradients in both projection matrices on this
  same multi-prefix batch. These were tiny controls, not pilot continuations.
- After that real update, BOTH arms saved and reloaded with correct mode and
  provenance; exact eval/inference logits matched on the length2 batch.
- Eight manifest tamper checks rejected schema, config hash, incomplete
  dependency map, state split, seed, update, protocol hash and schedule changes.
- A deliberately modified learned table with matching substituted artifact
  digest and manifest digest was rejected against reconstructed seed0 state.
- All118 protected files verified unchanged.

Direct-control script executed successfully from `/tmp/e17_review_controls.py`;
its temporary checkpoint/preflight directory was removed on completion. Future
maintenance may move these exact length2/all16 controls into committed tests.
That maintenance is not required before this narrowly reviewed pilot because
the controls above have actually run against the final source below.

## Frozen reviewed hashes

- module: `984301fe2174da2735f08deef0aa2eb1425664036698c7ffe412aa62d78cc160`
- runner: `471750dc55bad8f05f882971111a0c5ecbb4816362e830787de2ed6b4daa1649`
- tests: `d532331d86be31f006d4e1c842fcff4a24f5f4a5762780607f4bd1da09143be9`
- protocol: `f067b1f6fa92b50c0c72d8e017dbccc65981922f8689d0d3b49e6433dbf553e3`

## Scope and remaining scientific limits

Both arms retain the float core, four substeps, output classes, memory/loss,
seed0 stream and fixed2000 budget. Counts152768/151232 are intentional. Common
initial state is bitwise matched; input functions/geometry are not. The planned
comparison tests the combined representation and optimization constraint under
one seed/budget, not a unique mechanism or capacity-independent effect.

No blocking code finding remains for root's fixed CLI preflight/run path.
Training must stop on partial failure; recovery permits evaluation of two valid
final checkpoints only. Report all32 seen-program rows and review both trained
checkpoints independently; no new cases, extra seeds or automatic extension.
