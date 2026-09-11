# Latent state-transfer audit — 2026-09-11

## Scope

This was one inference-only audit of the final local-1000 checkpoints from the
gated-carry pilot. It ran 16 forwards over 256 states for each of four pairs of
semantically identity paths, followed by a common `ADD` probe. It performed two
checkpoint loads, zero backward passes and zero optimizer updates. Model and
adapter digests were unchanged after every audit forward, and the checkpoint
SHA256 values still match the launch gate.

The compared pairs were `SWAP SWAP` versus four `SWAP`s, `XOR XOR` versus four
`XOR`s, one mixed identity cycle versus two cycles, and `SWAP SWAP` versus `XOR
XOR`. The last pair tests path dependence between two different realizations of
the same DSL identity.

## Findings

The writer update is necessary: the earlier no-write path did not transfer
state. The current writer does carry a state forward, but its latent
representation is not semantically invariant. After an identity path, the
distance from the initial slots is large: mean relative L2 was about 0.938 for
A and 0.933 for B on the tested paths. This is a representation-drift signal,
not a claim that the output is always wrong.

For repeated cycles of the same operation, the two path lengths remain fairly
close but not identical. Mean relative L2 between the states entering the
common probe was:

| Identity pair | A | B |
|---|---:|---:|
| `SWAP SWAP` vs `SWAP`×4 | 0.0206 | 0.0245 |
| `XOR XOR` vs `XOR`×4 | 0.0151 | 0.0248 |
| mixed cycle vs two cycles | 0.00141 | 0.00205 |
| `SWAP SWAP` vs `XOR XOR` | 0.1988 | 0.2152 |

The cross-operation identity pair is much farther apart, showing that the
writer representation depends on the path even when the DSL state is the same.
The common probe's argmax predictions agreed between the two paths in all 256
states for every pair, although this probe was outside the registered training
scope and is diagnostic rather than a new accuracy claim. Mean absolute probe
logit differences were 0.0035–0.0556 for same-operation/mixed pairs and 0.3685
for the cross-operation pair in A; B was 0.0048–0.0695 and 0.3700.

Arm B therefore does not provide a state-preserving improvement over A. Its
same-operation pair distances are slightly larger, consistent with the
negative full-pool result. The gate is not the missing semantic state contract.

## Consequence

The next meaningful change is a writer transition constraint while retaining a
real `z_t → z_{t+1}` update. A cycle-consistency objective over registered
identity cycles is the smallest testable candidate: ordinary operations may
change `z`, while a semantic identity cycle is trained to return compatible
carry. It should first pass pure and tiny reload QA, then a short screening run;
no such training run was started here.

The audit script is [pc_state_transfer_audit.py](../scripts/pc_state_transfer_audit.py)
and the raw saved report is
[state_transfer_audit/report.json](../runs/pc_gated_carry_v1/state_transfer_audit/report.json).
The first technical failure is preserved in
`runs/pc_gated_carry_state_audit_background_v1`; the repaired replacement
completed under `runs/pc_gated_carry_state_audit_background_v2`.
