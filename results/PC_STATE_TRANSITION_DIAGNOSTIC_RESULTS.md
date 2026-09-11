# Writer-to-cache transition diagnostic result — 2026-09-11

The v3 run completed successfully after two pre-forward runner repairs. The
repairs changed only runtime bookkeeping and helper binding; the endpoint,
paths, states, measurements and budget stayed fixed. The v1 and v2 failed
trees remain preserved as setup-failure artifacts.

## Execution and coverage

The accepted local-2000 endpoint was loaded once. The diagnostic performed three
inference forwards: one for each path and one batched four-way substitution. It
recorded 12 cases, 80 readout positions and 640 native steps, with zero
backwards, zero optimizer updates and zero failures. The independent saved
artifact audit is
`runs/pc_latent_slots_v1/state_transition_diagnostic_v3/saved_audit.json`.

The selected pair remained the frozen `padding_ADDADD_ADD_k4` good path and
`padding_ADDADD_ADD_k10` bad path at states `(6,4)` and `(8,14)`. Their semantic
states immediately before the final `ADD` match. The fresh path replay gives
good `2/2` correct and bad `0/2` correct, matching the saved expectations.

## What differs at the transition

The two final slot inputs differ materially (mean relative L2 `0.3945`), and
the difference remains in the reader sum `h` (`0.4023`), reader output `v`
(`0.3783`), normalized values (`0.4474`) and projected values (`0.4147`). The
output-normalized writer input `phi` differs more strongly (`0.7640` relative
L2), with a mean writer-output delta norm of `7.7529`.

The native attention rows do not collapse in the bad path. Near-decisive rows
are `0.7031` for good and `0.7188` for bad; near-uniform rows are `0.0156` in
both. Final attention mean absolute differences between paths are only `0.0191`
for `(6,4)` and `0.0112` for `(8,14)` under the saved row layout. This points
away from a simple role-key absence or uniformly random reader selection.

## Four-way substitution

At the common final `ADD`, the two cached key/value sets determine the answer
in both states:

| h source | KV source | correct cases |
|---|---|---:|
| bad | bad | 0 / 2 |
| good | good | 2 / 2 |
| good | bad | 0 / 2 |
| bad | good | 2 / 2 |

Holding KV fixed makes the prediction unchanged when only `h` is swapped. In
this bounded substitution, the discriminative information is therefore in the
path-dependent cache KV state. That localizes the next hypothesis to state
preparation, writer output and KV formation together; it does not isolate the
writer or LayerNorm as the sole cause.

A read-only factorization of the saved final arrays sharpens this result. The
projected keys are exactly equal across good and bad at both states (maximum
absolute delta `0.0`), while projected values differ (`L2 1.0655` and
`1.3674`). The combined KV substitution is therefore effectively a V-only
substitution for this pair. No additional model forward was needed; the saved
factorization audit is
`runs/pc_latent_slots_v1/state_transition_diagnostic_v3/kv_factor_readonly_audit.json`.

## Decision boundary

This is a completed observational diagnostic, not a causal proof and not a
model change. It does not establish global non-Markov behavior, universal
failure, or a LayerNorm-specific mechanism. Stop here. Any next implementation
or experiment must state a new writer/state-preparation mechanism, preserve the
existing handoff and frozen inputs, and pass a fresh bounded gate before model
work.

Saved run files:

- report: `runs/pc_latent_slots_v1/state_transition_diagnostic_v3/report.json`
- accounting: `runs/pc_latent_slots_v1/state_transition_diagnostic_v3/accounting.json`
- good path: `runs/pc_latent_slots_v1/state_transition_diagnostic_v3/good_path.json`
- bad path: `runs/pc_latent_slots_v1/state_transition_diagnostic_v3/bad_path.json`
- substitutions: `runs/pc_latent_slots_v1/state_transition_diagnostic_v3/substitutions.json`
- input freeze: `runs/pc_latent_slots_v1/state_transition_diagnostic_v3/input_freeze.json`
- V-factor audit: `runs/pc_latent_slots_v1/state_transition_diagnostic_v3/kv_factor_readonly_audit.json`
