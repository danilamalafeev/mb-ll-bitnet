# Projected-value cycle screening protocol

## Purpose

This is one bounded screening of the accepted projected-value identity-cycle
contract. It asks whether the auxiliary loss improves the measured
projected-V consistency of the existing latent-slot writer while preserving
the ordinary pointer-chasing trace.

The screening compares two arms that start independently from the same
accepted E36 endpoint and deterministic adapter initialization:

- `ordinary`: accepted supervised latent-slot loss only;
- `cycle_loss`: the same loss plus the accepted detached projected-V cycle
  loss, weight `0.1`, using the native `reader -> memory_norm -> v_proj` path.

The writer, reader, cache construction, model width, optimizer and native
step count are unchanged. No external state is copied, no KV is injected and
no previous screening result is resumed.

## Frozen scope

Each arm runs the first 64 batches of the immutable B training stream, with
batch size 64 and the existing length schedule
`(1, 2, 3, 4, 5, 6, ...)`. The prefix contains 64 updates, 4,096 examples,
14,080 readout positions and 112,640 native steps per arm. The prefix has
1,225 detected identity-cycle windows across 53 updates; only the cycle arm
applies that auxiliary term.

Evaluation is performed before and after training for the following eight
predeclared programs:

`padding_ADDADD_ADD_k4`, `padding_ADDADD_ADD_k10`,
`padding_ADDADD_XOR_k4`, `padding_ADDADD_XOR_k10`,
`padding_XORSWAP_ADD_k4`, `padding_XORSWAP_ADD_k10`,
`composition_L12_0`, `composition_L16_0`.

For each program, only the fixed validation and test state strata are used
(64 states). This gives 8 forwards, 512 cases and 8,704 positions per
arm/phase. Projected-V cycle loss is recorded during evaluation as a
diagnostic for both arms; it is not optimized during evaluation.

## Registered budget

| Quantity | Budget |
|---|---:|
| endpoint loads | 2 |
| underlying deserializations | 4 |
| training forwards | 128 |
| evaluation forwards | 32 |
| total forwards | 160 |
| total cases | 10,240 |
| total readout positions | 62,976 |
| total native steps | 503,808 |
| backwards / optimizer updates | 128 |
| committed final checkpoints | 2 |
| training cycle windows seen | 2,450 (1,225 per arm) |
| cycle-loss updates applied | 53 (cycle arm only) |
| evaluation cycle windows recorded | 12,544 |

All counters are attempted/completed counters. A nonzero failure, incomplete
arm, changed source binding, changed stream prefix, changed state selection or
changed accounting fails the run.

## Screening criteria and stop

The primary diagnostic is the final held-out weighted mean raw projected-V
cycle loss. A cycle arm reduction of at least 5% is the registered positive
screening threshold. The task guardrail is final full-trace correctness: the
cycle arm must not lose more than five percentage points against ordinary on
the same 512 held-out cases. A result that meets neither condition is
negative for this bounded parameterization; a result that meets one condition
is inconclusive and does not authorize a larger run.

This protocol makes no superiority, causal or universal-correctness claim. It
does not authorize a full pilot, tuning, retries, additional probes or science
after completion. Any next experiment requires a new explicit decision and a
new source gate. All failed attempts and their costs remain preserved.
