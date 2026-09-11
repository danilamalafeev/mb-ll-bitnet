# State-aware carry protocol — 2026-09-11

## Question

The accepted latent-slot path must write a new state after every operation. The
state-transfer audit shows why the current transition is unstable: the reader
creates two slot-specific hidden values, then `fresh_cache_from_slots` sums
them into one `h` before the ordinary writer sees it. The writer therefore
has to reconstruct two slot vectors from an aggregate and cannot make the
transition Markov in the carried state. The earlier no-write path is not a
control for this question because it transferred no state at all.

This experiment tests the smallest structural repair: retain the ordinary
writer exactly as a base path, and add a zero-initialized correction that sees
the current flattened slots and returned hidden vector:

```text
z_(t+1) = W_base(h_t) + C([z_t, h_t])
```

The correction has `Linear(160, 32)` shape and 5,152 parameters. At its zero
initialization the new path is exactly the accepted ordinary writer, so any
change after training comes from the added state input rather than a changed
endpoint or initialization. No cycle-specific loss, opcode-specific input,
target input, KV bypass, or external state copy is allowed.

## Arms and lineage

Arm **A** is the accepted ordinary latent writer loaded from the immutable
local-2000 endpoint. Arm **C** loads the same endpoint and wraps only its
writer with the state-aware correction. The old writer parameter objects stay
inside `base_writer`; its loaded optimizer moments remain attached to those
objects. The two correction parameters are appended to the existing adapter
optimizer group with empty optimizer state. Both arms use the same frozen
accepted training stream and native-eight forward path. Historical latent and
gated artifacts are read-only inputs.

The accepted latent helper remains the forward owner. A scoped reader hook
captures the current `z_t`, and a scoped writer hook supplies `[z_t, h_t]` to
the wrapper. Hooks are removed on success and every exception path. The
runtime must record the wrapped adapter identity and optimizer name/order
separately from the accepted ordinary identity.

## Gates

1. **Pure contract.** Test the 160-wide input, exact zero initialization and
   ordinary-output equivalence, state sensitivity, finite/shape rejection,
   hook pairing and cleanup, preservation of base parameter objects and
   optimizer moments, correction-only optimizer state, and immutable wrapped
   identity. No model or checkpoint work is permitted at this stage.
2. **Tiny QA.** Load the accepted endpoint independently for A and C. Run the
   existing two-example L1/L2 snapshot-resume contract for three updates per
   arm (six forwards, 12 cases, 20 positions, 160 native steps, six backward
   passes, six optimizer updates, two snapshots and eight underlying
   deserializations). Require exact uninterrupted-versus-reloaded model,
   wrapped-adapter, optimizer, CPU/CUDA RNG, mode and parameter-name equality.
3. **CODE CLEAR and source freeze.** Check the changed helper, tests and QA
   runtime against their actual SHA256 values and the pure/QA evidence. Do not
   re-run accepted historical QA merely to reorder the gates.
4. **Bounded screening.** From the accepted local-2000 endpoint, train A and C
   for the first 250 fixed stream batches each (500 updates total), then
   evaluate the registered old 69 programs plus the 81 identity/composition
   controls used by the gated pilot (150 forwards per arm). Add the existing
   16-forward state-transfer audit at the endpoint of each arm. This is a
   screening boundary, not a full 2,000-update claim. Stop after the saved
   screening report and independent artifact verification.

## Acceptance and stop

The screening is useful only if C reduces the state-transfer inconsistency
signal (same-operation identity-path distances and/or cross-operation path
dependence) while avoiding substantive regressions in registered full-trace,
padding, composition, and identity-control accuracy versus A. A result that
does not meet both conditions is a negative structural screening and stops;
do not tune the correction, add a cycle loss, or retry automatically. A
screening that meets both conditions may motivate a separately registered
full pilot; it does not authorize that pilot by itself.

Every stage writes to a fresh `runs/pc_state_aware_carry_v1/` subtree and
preserves the prior latent, pair-carry, gated-carry and state-transfer
artifacts. Long QA/screening execution uses one detached `run_and_wake`
supervisor per registered stage. No science interpretation occurs before QA
acceptance and source CODE CLEAR.
