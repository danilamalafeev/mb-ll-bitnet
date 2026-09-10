# Latent slots padding-drift diagnostic protocol

2026-09-10. Prospective design only. This document registers one later disposable diagnostic run; this turn performs no checkpoint load, model construction, forward, training, QA, or science execution.

## Question and attribution

The accepted latent pilot reaches 17,628/17,664 full traces. Its remaining 36 errors are all padding cases at lengths 24 and 32: two states for each of the nine padding programs at each length. The first errors occur at positions 21 or 22 and none recover. The diagnostic asks whether the failure is associated with accumulation during the repeated padding extension, a particular slot or channel, or an abrupt transition. It is an observational follow-up to the saved final endpoint and cannot establish that a measured norm causes an error.

The report must keep `padding_extension` and `useful_op` errors separate. `padding_extension` is a role assigned from the frozen padding program identity and position map; it is not a claim that an individual `SWAP` is semantically a no-op. Useful operations are the remaining prefix and suffix positions. No error is reclassified from its true DSL target or decoded output.

## Fixed run and budget

Use the saved latent endpoint at `runs/pc_latent_slots_v1/science/checkpoints/local2000.pt`, with its exact source, manifest, parent lineage, adapter identity, optimizer state, and runtime settings. The scope is the same 69 saved programs and all 256 states, in the saved program/state order and train/validation/test strata. Strata must use the accepted E15 hash-ranked split, `sha256("E15-state-v1:x:y")` ordered by hexadecimal digest with 192 train, 32 validation, and 32 test states; the lexicographic `STATE_ORDER` remains the iteration order. The bound implementation is `looped_bitnet/register_e15.py` (SHA256 `bf3e7fa45dfaf2226cd8b1e4541afce36d797ac69f8ad7c0a975eba041561dac`). The detailed diagnostic rows are required for the 18 padding programs at lengths 24 and 32 (nine at each length); the other 51 programs still run in the same observational evaluation so the endpoint scope and total accounting remain comparable.

The baseline budget is 69 whole-program forwards, 17,664 cases, 331,776 instruction positions, and 2,654,208 native steps. The focused subset is 18 forwards, 4,608 cases, 129,024 positions, and 1,032,192 native steps. There are zero updates, backwards, or optimizer steps. Counts are actual attempted and completed values, persisted before and after each forward. Parent load and every underlying deserialization are counted independently. A failure stops the run and preserves the partial accounting; there is no retry or resume mode.

The first run is the observational baseline. A writer-frozen, zero-write, or slot-carry-forward control may be registered later only if the accepted implementation exposes that behavior without changing the normal interface or bypassing its cache semantics. Such a control is optional, receives its own immutable mode and output directory, and is not included in the baseline budget. If approved, each control is limited to the 18 focused forwards (4,608 cases, 129,024 positions, 1,032,192 native steps) unless a new protocol explicitly registers the full 69-program scope. No control run is promised by this design.

## Measurements

For every focused program/state and every instruction position, save:

- immutable program ID, suite, length, opcode, state, and stratum;
- the true DSL target pair and decoded pair, exact pair error, x-error, and y-error;
- the explicit role (`padding_extension` or `useful_op`);
- one-based first error, first subsequent recovery, final correctness, full-trace correctness, and recovered-final flag;
- two slot-z L2 norms, two slot-z L2 deltas, and signed-free per-channel absolute z deltas for each slot;
- two writer-output L2 norms, two writer-output L2 deltas, and per-channel absolute writer deltas for each slot;
- hidden/output norms and finite flags, plus a compact per-position finite result.

The first position has no previous-position delta and records `null` for delta fields. The writer at position `t` is the next instruction's input `z[t+1]`; its shift is checked explicitly and writer deltas are labeled as a shifted view, never independent same-position evidence. The diagnostic stores these reduced values and decoded/target traces, never a full hidden-state or KV dump. Norms and slot/channel diagnostics are descriptive measurements, not bit accuracy, semantic slot labels, or expected-bit supervision.

The helper freezes the exact 18 selected padding IDs, opcode sequences, and position roles. The eventual runner derives the same position map once from each saved padding program ID and exact token sequence, stores it in the input manifest, and checks every recorded opcode, role, state, and `writer[t]`/`z[t+1]` join before model construction. A program/state join, target trace, or position/opcode mismatch aborts before any forward.

## Immutable inputs and output safety

Before model work, create a fresh `runs/pc_latent_slots_v1/padding_diagnostic_v1` directory and refuse a nonempty destination. Freeze and SHA-256 bind the following actual bytes:

1. the accepted latent science manifest and digest sidecar;
2. `local2000.pt`, its checkpoint metadata, and the accepted final evaluation bytes;
3. the frozen evaluation scope, stream/program identity, state-stratum split, and saved baseline lineage;
4. the complete transitive local source inventory already accepted for the latent pilot, including the strict loader/adapter and numerical helper;
5. this protocol, the diagnostic helper and focused tests, and the exact runtime configuration.

The runner must compare current bytes to the frozen inventory before constructing a model or deserializing the endpoint. It must validate the checkpoint's local update 2000, absolute update 42000, architecture ID, adapter parameter names/order, and source/manifest lineage. Generated accounting and diagnostic outputs are outside the immutable input manifest. JSONL/raw rows, aggregate JSON, input freeze, and final failure status are committed once with same-volume flushed temporary files followed by atomic rename and refuse overwrite. The mutable latest accounting snapshot is also written through a same-volume flushed temporary file and atomic replacement so a partial attempt is durable; that replacement does not alter the immutable manifest, accepted evidence, or any previously committed diagnostic row.

## Aggregation and interpretation

The saved report includes per-program rows and aggregates by length, stratum, and length×stratum, with separate padding-extension/useful-op error totals, first-error histograms, first-subsequent-recovery histograms, exact final/full denominators, and finite-failure counts. The all-69 marginal counts are retained for scope accounting, while the 18-program focus is the diagnostic conclusion set. Controls, if later registered, are kept as separate candidate modes and never merged into the baseline.

Use the following predeclared descriptive checks:

- **Monotonic accumulation:** slot-z norm is nondecreasing over positions and cumulative per-position z deltas rise through the padding extension, with the first error occurring in or after that extension. This is an association check, not a causal test.
- **Single slot/channel concentration:** report the dominant slot and channel using cumulative absolute z-delta L1 totals. The channel fraction denominator is the total L1 delta within its slot. Mark concentration when the largest slot or within-slot channel contributes at least 75% of the relevant nonzero total; retain totals and fractions, and report zero-total concentration as explicitly unavailable (`None` in the pure helper and JSON `null` in a report), rather than as a negative result.
- **Abrupt threshold:** use the total z-delta L1 across all 32 slot channels at the first error position. Report a first-error-side jump only when at least four valid preceding delta values exist and that delta is at least four times their median. If the median is zero, a positive first-error-side delta qualifies and a zero delta does not; otherwise report that the threshold test is unavailable. Record the first-error role and compare errored versus correct-position L1 totals within each frozen role.
- **Useful-operation separation:** compare the error totals and first-error positions by explicit role. A padding-only pattern supports a padding-drift description; any useful-op errors remain visible and prevent that description from being generalized.

If a control is eventually run, compare decoded/DSL traces and the same reduced diagnostics at identical program/state/position joins. A control can localize an implementation effect only if its semantics are independently reviewed and its exact mode, source bytes, and costs are bound. No control outcome may be described as proof of a learned representation or as a semantic explanation of slot channels.

## Pure gate and future execution boundary

The pure helper `scripts/pc_latent_slots_padding_diagnostic.py` validates the immutable 18-program focus/opcode/role map, contiguous step alignment, finite compact z/writer summaries, explicit writer-to-next-z shifting, true-target trace errors, first-error/recovery, L1 concentration and role/error association, length×stratum/program aggregates, and the registered median×4 cumulative-drift oracle. Its tests use hand data and the DSL target constructor only; they do not load checkpoints or construct the model.

Independent source review must clear this protocol, helper, tests, and the exact frozen input plan before any later runner is written or launched. The only eventual command shape is a separately reviewed diagnostic entry point with an explicit fresh output directory; no command is authorized by this design.
