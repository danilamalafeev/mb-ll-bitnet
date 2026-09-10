# E18 pre-training code review — HOLD

Independent reviewer: Astra/low, 2026-09-07. This records the initial inspected
implementation before executor repairs; no E18 scientific training is cleared.
Scope: new E18 module, runner and focused tests against the registered protocol.
Source was still being finalized, so this is a preserved interim finding record,
not a hash-frozen clearance. A later FINAL review must identify stable hashes.

The recurrence implementation appears faithful: both budgets use the same
four FFNs, inject opcode each substep, carry cache/state and read out once per
instruction. Native4 parity uses explicitly asserted length2 stream index667
and compares logits, loss, all gradients and one actual clipped AdamW update.

Blocking gaps in the initial inspected version:

1. Checkpoint `internal_state_updates_per_arm` and its loader expectation used
   2000*64*B, missing the stream's mean instruction count2. It must be1024000
   for4 and2048000 for8, consistent with manifest/runner training accounting.
2. Recovery omitted `_initial_states`, despite the registered requirement to
   validate saved initial states during recovery. It returned evaluations and
   pairing only, omitting the full recoverable report's rows, deltas, costs and
   provenance. Main final reporting also omitted inference-work accounting.
3. The native8 explicit-unroll control exercised only the first instruction.
   Its backward check tested nonzero gradients but did not compare gradients to
   explicit unrolling. Required multi-instruction carry/KV/counter, block order
   and exactly one boundary readout checks must be demonstrated together.
4. Manifest validation omitted its explicit format, encoder mode and parameter
   count fields. Validate these existing claims against registered expectations.

These are narrow repairs, communicated to the executor. No architecture change,
new scientific evaluation, extra seed or expanded experiment is requested.
Targeted tests were started as verification; their final status belongs in the
later stable-source review. HOLD remains until the registered controls pass on
stable sources and the independent final review clears them.
