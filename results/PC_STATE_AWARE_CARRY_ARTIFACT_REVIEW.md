# State-aware carry post-screening artifact review — 2026-09-11

This is a read-only review of saved artifacts. It performed no model forward,
backward pass, optimizer update or checkpoint replay. The machine-readable
record is `runs/pc_state_aware_carry_v1/posthoc_artifact_review.json`.

## Ordinary continuation resolves the old finite-set failures

The original latent science final evaluation covered 69 programs and 256
states per program, or 17,664 cases. It had 36 full-trace failures, all in the
registered padding family and first appearing at position 21 or 22.

The screening's ordinary arm A loaded the same accepted local-2000 endpoint and
then trained for 250 updates. Its first 69 final rows use the same program and
state IDs as the old evaluation and contain 17,664 / 17,664 full-trace correct
cases. The 36 old failure IDs and states were matched exactly and none remains
wrong after this ordinary continuation. A also reached 20,736 / 20,736 on the
new identity controls.

This is a finite-set continuation result, not a proof of universal correctness.
It does mean that the old 36 cases cannot be used as evidence that the ordinary
architecture is structurally unable to solve this evaluation.

The continuation used the exact first 250 batches of the accepted 2,000-batch
`science_stream.json`: batch size 64, cyclic lengths, total length 871, saved
prefix digest `620662e4c0d89bbd1391d27be1f2e8124ff254d6c20203c34542490e91ea7f31`.
"Frozen stream" therefore describes the data stream; model and adapter
parameters were updated in both arms.

## C starts as an exact parameter extension, then follows a different training path

The saved local-0 checkpoints show, under the ordinary-to-base-writer name
mapping:

- all parent model tensors are equal;
- all ordinary adapter tensors equal C's `base_writer` tensors;
- C's correction weight and bias are exactly zero;
- old base optimizer states are equal and the correction state is empty;
- both optimizer groups use `lr=0.001`, Adam betas `(0.9, 0.999)`,
  `eps=1e-8` and `weight_decay=0.01`.

This is a static checkpoint comparison; no forward replay was needed or run.
The later C result therefore rejects this particular state-aware
parameterization together with its 250-update continuation, not the general
idea that writer access to current state could be useful. C trains the original
model, the original adapter/base writer and the new correction parameters
together. Its correction adds a direct state path and a second hidden-to-slot
projection, so the experiment is not an isolated access-only intervention.

## Interpretation boundary

The implementation contains `values[:, 0] + values[:, 1]` when constructing the
initial hidden vector, but the core also retains separate slot-specific values
in its role-keyed KV input. The aggregate hidden path is therefore a plausible
bottleneck hypothesis; it does not establish that the complete transition is
globally non-Markov or that the writer alone is the cause.

Likewise, latent identity-path distance is diagnostic. A has mean relative L2
0.0613 and perfect saved traces, so latent distance alone is not a correctness
criterion.

The screening remains a valid negative result for C: C reached 21,085 / 38,400
full traces, had 8,549 old-program and 8,766 control regressions, and worsened
the identity-path diagnostics. No full C pilot, tuning, cycle loss or retry is
warranted from these artifacts. The frozen protocol and completed screening
outputs remain unchanged; any continuation requires a newly stated mechanism
and a fresh bounded gate.

Inputs and hashes are recorded in the JSON review artifact. The original
screening report and independent saved audit remain the authoritative execution
records.
