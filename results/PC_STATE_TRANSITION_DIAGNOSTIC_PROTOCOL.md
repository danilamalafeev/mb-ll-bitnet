# Writer-to-cache transition diagnostic protocol — 2026-09-11

## Purpose

Test the concrete hypothesis that a writer output is not robustly recovered by
the next reader/cache path. The diagnostic records the actual transition
`phi -> z -> v -> memory_norm(v) -> projected V`, captures all eight native
attention weight rows, and performs a four-way diagnostic substitution of the
initial working vector `h` and cached key/value tensors.

This is a bounded observational experiment. It does not train, resume, tune or
modify the model architecture or any checkpoint.

## Frozen source and endpoint

- Endpoint: accepted latent `local2000` (`absolute_update=42000`), one seed,
  float width-128 arm.
- Good path: `padding_ADDADD_ADD_k4` (12 operations).
- Bad path: `padding_ADDADD_ADD_k10` (24 operations).
- States: `(6,4)` and `(8,14)`, the two saved old-failure states for the bad
  path. The two paths have the same semantic state immediately before their
  final `ADD`.
- Saved final evaluation must still mark good states correct and bad states
  wrong before any endpoint load.

## Measurements

For both paths and both states, every position stores the slot input `z`,
reader output `v`, summed `h`, normalized values, projected keys/values,
returned hidden state, `phi=output_norm(returned_h)`, writer output and all
eight native-step attention weights. The report reduces final-position pairs
and records the complete bounded path files.

At the common final operation, one batch evaluates:

1. bad `h` + bad KV;
2. good `h` + good KV;
3. good `h` + bad KV;
4. bad `h` + good KV.

The substitution is a diagnostic input replacement only. It is not a proposed
architecture and must not be interpreted as causal proof by itself.

## Exact budget and stop

The run contains two path forwards and one batched substitution forward:
12 cases, 80 readout positions, 640 native steps, zero backwards and zero
optimizer updates. One endpoint load is allowed through the accepted loader.
The output is a fresh `runs/pc_latent_slots_v1/state_transition_diagnostic_v1`
subtree with accounting, input freeze, two path files, substitution outcomes
and one report. Stop after one saved-artifact inspection; no retry, training,
cycle loss or broader evaluation is authorized by this protocol.

Any interpretation must remain limited to the tested endpoint, paths and
states. A result that implicates KV, `h`, or their combination identifies a
location for the next hypothesis; it does not establish that LayerNorm or the
writer alone caused the original failure.
