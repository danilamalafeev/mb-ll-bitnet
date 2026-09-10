# E17 independent code review — initial HOLD

Reviewer: Astra/low, 2026-09-07. Scope: new E17 module, runner and tests against
E17 protocol. No registered pilot training performed. Initial targeted suite:
7 passed in6.09s using `.venv/bin/python -m pytest -q tests/test_bit_input_e17.py`.
Passing tests do not clear the following missing guards/controls.

## Findings requiring bounded repair

1. Root identified that load_checkpoint compared trained state to initial and
   common-initial digests after load, rejecting trained checkpoints. A fix
   arrived during review: reconstruct and validate initial digests BEFORE load,
   then validate final model digest only. Require actual one-update save/load
   regression controls for BOTH arms, with exact eval logits. Root also
   requested checkpoint environment metadata; its addition is now visible.
2. Runner `_initial_states` reconstructs the registered seed0 digests but ignores
   those expected values. It only validates loaded states against the manifest.
   Compare all three manifest initial/common digests to reconstructed expected
   digests before loading states, so a self-consistent substituted initial
   artifact cannot reach training.
3. `_load_manifest` does not validate the complete frozen programs/state_split,
   schedule, seed, update, top-level protocol/config hashes or exact dependency
   map. Later coverage validates train programs/states, but a bad validation
   set is rejected only after training. Check these fixed manifest fields and
   complete source sets before update1 against the frozen source/specification.
4. Source hashes omit `looped_bitnet/model.py`, the imported core architecture
   implementation. Add this runtime dependency to the E17 frozen map. Existing
   protected-file checks reduce drift risk but are not a substitute for the
   required manifest dependency provenance.
5. Complete focused controls: exercise both actual SignedBitsEncoder forward
   maps over all16 inputs against explicit signed-feature matmul; use unequal
   marginal totals in the paired toy (current2vs2 makes signed delta trivial);
   add no-training recovery control with both checkpoints and no report, and
   missing-arm refusal. Include tamper cases for repaired initial/manifest
   guards and schema/config hashes. These tests must remain tiny controls.

## Confirmed design paths

Signed bits are float32 LSB-first, range/type checked, with local seed0 x-then-y
projection draws at std0.01 and fork_rng-protected module construction. Common
parameters are cloned and compared bitwise with disjoint storage. Counts match
152768/151232; both encoders have no bias and no codebook remains. Existing
init_cache routes their vectors into h and memory. E16 learned logits, gradients
and one update reproduce exactly in the current targeted suite. E16 stream and
target digest checks are reused. Runner uses independent frozen-config AdamW,
finite loss/gradient checks, fixed2000 stream, and atomic final checkpoints
before evaluation. Evaluation scope and semantic paired field names are correct.
Recovery code has no training call. Old sources are imported, not edited.

Training remains HOLD until a separate final review checks repaired stable
source and tests. Preserve this initial review; do not replace it with CLEAR.
