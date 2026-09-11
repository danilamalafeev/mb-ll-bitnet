# Projected-value cycle contract — 2026-09-11

## Question

The completed transition diagnostic found that the two failing paths have
identical projected keys but different projected values at the common final
operation.  The ordinary writer is still required: the earlier no-write path
did not transfer state.  The next bounded mechanism therefore keeps the
existing `z_t -> z_(t+1)` writer update and adds a loss only when a program
contains a semantic identity cycle.

For a detected identity cycle, the target is the per-slot projected value
memory used by the next cache:

```text
V(z) = v_proj(memory_norm(reader(z)))
L_cycle = mean( ||V(z_after) - stopgrad(V(z_before))||²
                / max(mean(V(z_before)²), eps) )
```

The target is detached so the objective preserves an earlier state while
training the post-cycle transition.  Ordinary operations keep the accepted
cross-entropy objective and ordinary writer.  There is no key change, direct
KV injection, external state copy, opcode-specific writer, or architecture
replacement.

## Pure gate

`scripts/pc_value_cycle_contract.py` and
`tests/test_pc_value_cycle_contract.py` are the only new implementation files
at this stage.  The pure gate checks exact reader projection parity with the
native cache, disjoint detection of `SWAP SWAP`, `XOR XOR`, and
`SWAP XOR XOR SWAP`, finite/shape rejection, a differentiable post-cycle loss,
detached targets, and the zero-work path when a batch has no identity cycle.
It constructs no checkpoint or training model.

## QA gate

One bounded QA run loads the accepted latent local-2000 endpoint once and
applies the contract to two fixed two-example identity batches: `SWAP SWAP`
and `XOR XOR`.  It performs three updates (first batch, second batch, and the
same second batch after snapshot reload), one arm, one committed snapshot,
three forwards, six cases, twelve readout positions, 96 native steps, three
backward passes and three optimizer updates.  The run must prove exact
uninterrupted-versus-reloaded equality for model, adapter, optimizer, RNG,
training modes and parameter association, and must record at least one cycle
window with a finite positive weighted loss.

The QA runtime may reuse the accepted endpoint loader and serialization
helpers, but it must bind the new helper, tests, protocol and runtime by their
actual hashes before loading the endpoint.  It writes only a fresh
`runs/pc_projected_value_cycle_v2/qa` subtree.  The accepted E36 loader performs
two endpoint-related deserializations (the endpoint and its parent-reference
checkpoint), so complete QA accounting is three underlying deserializations
including the committed snapshot reload.  No scientific evaluation,
screening, tuning, retry, or interpretation is part of this gate.

## Decision boundary

QA acceptance only establishes that the projected-value contract is wired
correctly and resumable.  A later screening, if explicitly chosen, must
compare ordinary training with the cycle-loss arm on the frozen stream and
measure both registered accuracy and identity-path projected-V consistency.
Failure of either condition stops the candidate; no automatic weight or
coefficient sweep is allowed.
