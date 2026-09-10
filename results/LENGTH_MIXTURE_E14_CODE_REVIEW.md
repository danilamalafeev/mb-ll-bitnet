# E14 length-mixture pretraining — independent pre-code review

**Status: PENDING CODE.** This is a bounded feasibility review performed before
inspection of the E14 implementation. No E14 training was run and no existing
core, script, run, or protected artifact was changed by this review.

## Scope and feasibility

The registered protocol in `docs/RESEARCH_LOG.md` is feasible with the existing
structured-reader model and data helpers, subject to the implementation checks
below. The relevant E13 corrected baseline is concrete and reproducible:

- `configs/cycle16_two_hop_steps8_structured_reader.json` specifies the
  structured QAT model (`d_model=64`, `d_ff=256`, four blocks, four heads,
  152,512 parameters), cycle16 data, batch 64, 2,000 updates, AdamW, clipping
  1.0, four CPU threads, and deterministic execution.
- Each selected E13 corrected baseline report records update 2,000, 128,000
  examples, 1,024,000 fixed eight-step example-steps, the registered objective
  `final_ce_plus_step4_ce_centered_l2_cycle_boundary`, and a selected
  `checkpoint_best.pt` SHA-256. The three baseline reports also contain paired
  initial-state and data-stream digests.
- `SampleStream.sample(hops=2)` produces the E13 stream's table, start, and
  order using the same RNG path. `Example`/`encode` keep the answer out of the
  input; `collate` supplies targets as a separate tensor.
- The longest E14 condition is 16 encoded edges plus START/object, eight STEP
  tokens, and END: 59 tokens, within the model's `max_tokens=64`. The model's
  structured encoder counts STEP tokens and the recurrent loop can run explicit
  budgets beyond the default eight steps, so h3 training and h8 evaluation are
  representable without changing the architecture.

The protocol's causal interpretation remains limited as registered: the mixture
arm changes the training length/label schedule and therefore changes the
supervision package, while architecture, optimizer, initialization, and total
example-step budget stay fixed.

## Conditions that must be visible in the code

### Stream and initialization pairing

1. Every mixture arm must call `seed_everything(seed, ...)` before constructing
   both `ReasoningModel` and `SampleStream`, with the E13 seed and `seed+100000`
   stream seed. The initial model digest must be recorded.
2. For every update, first sample exactly the E13 h2 examples, e.g.
   `stream.sample(hops=2)`, then derive the homogeneous h1, h2, or h3 batch by
   changing only `Example.hops`. Calling `sample(hops=h)` on a variable schedule
   is unsafe if it changes RNG consumption or if separate streams are used.
3. The base-h2 digest must be computed over the same canonical collated h2
   batches and targets as E13 and must equal the corresponding full E13
   baseline digest. The actual mixture digest must separately include the
   variable STEP inputs, variable targets, and the exact schedule. It is invalid
   to label the actual mixture input digest as equal to E13 merely because the
   underlying tables/start/order are shared.
4. Because `collate` right-pads to the longest row in each batch, hashing
   mixture batches directly cannot by itself reproduce the fixed-h2 E13 digest.
   The implementation must either hash a separately collated h2 view of each
   source batch or use an explicitly documented per-example canonical encoding;
   it must then test byte-level digest equality.

### Schedule, objective, and gradient equivalence

1. Updates 1..1998 must be the exact repeating h1/h2/h3 schedule (666 of each)
   and updates 1999..2000 must be h2, giving h1=666, h2=668, h3=666,
   `sum(h)=4000`, 128,000 examples, and 1,024,000 example-steps per arm.
   Record and hash the complete schedule; do not infer it later from counts.
2. A homogeneous h batch must run exactly `4*h` recurrent steps and supervise
   every readout `4*k`, `k=1..h`, with targets `f^k(start)`. The loss must be
   `(2/h) * sum CE(readout_4k, f^k(start))`, including gradients from all
   readouts. For h2, this must be numerically and gradient equivalent to the
   E13 baseline loss (`CE(step4)+CE(step8)`) on the same initial state and batch.
3. The forward path must never receive a target, target-derived state, teacher
   forcing signal, detached prediction, or feedback. Labels are for loss and
   evaluator metrics only. A targeted causal-separation test should alter labels
   while holding input IDs fixed and verify that the forward logits/gradients
   before loss construction are unchanged.
4. Readouts must be taken from the live recurrent state at the requested step,
   with no post-readout mutation or accidental final-readout overwrite. Prefix
   semantics should be checked against a direct run at the shorter budget.

### Validation selection and gate

1. Build one fresh 256-record validation manifest with a shared table/start/order
   per paired h1/h2/h3 condition. It must be table-disjoint from historical
   validation sources and have a pre-training fingerprint. Separate unpaired
   validation streams for each h would change the paired design.
2. Checkpoint selection is by **macro** final accuracy over native h1@4,
   h2@8, and h3@12 (mean of the three stratum accuracies), then mean native
   final CE. It must not use a pooled count with a hidden length weighting or
   any test result. The tie rule must be deterministic and recorded; the E13
   convention of retaining the earliest update on an equal score is suitable.
3. Validation cadence must be exactly every 250 updates through update 2,000.
   The seed-0 gate has six cells, namely every native readout in the three
   strata: h1@4; h2@4 and h2@8; h3@4, h3@8, and h3@12. Every cell must be at
   least 95% accuracy at the selected checkpoint. Do not reduce this to only
   three final cells or to a pooled accuracy.
4. If seed 0 fails, stop after seed 0 and evaluate only that mixture arm and its
   matched historical baseline. If seed 0 passes, run seeds 1 and 2 to the
   fixed budget regardless of their own gate values. The gate must be resolved
   before any test logits are produced.

### Baseline reuse and evaluation

1. The historical controls must be exactly the E13 corrected baseline
   `checkpoint_best.pt` files for seeds 0/1/2. Before loading, independently
   validate the report SHA, diagnostic tag, objective, dynamics, config, seed,
   update=2000, parameter count, and `resume_supported=false`; also verify the
   paired initial/data digests and the protected preflight snapshot. Loading an
   untagged or latest/resumable checkpoint must fail closed.
2. Test evaluation must use one new 512-table manifest shared by all arms and
   explicitly evaluate input h1..h8 at budgets 4,8,...,32. Baseline and mixture
   must receive the same encoded conditions and metrics; targets are consulted
   only after predictions. The primary h8 comparison therefore means native
   h8@32 for both checkpoints, not E13's native h2 validation condition.
3. Test manifests and logits must be created/evaluated only after all permitted
   training and gate decisions. Save per-hop predictions, counts, and paired
   wins/losses so the registered primary and strong predicates can be audited.

### Novelty and artifact protection

1. Novelty checks must fail closed for every mandatory source: persisted E07
   `eval_sets`, reconstructed E07 suites, E08 pairs, E09, E10, E11, E12,
   invalid E13, and corrected E13. Missing files, malformed top-level/schema,
   empty records, malformed records, and missing referenced E10 manifests must
   raise an error. Silently skipping malformed entries is insufficient.
2. Compare canonical transition tables, require uniqueness within each new
   manifest, verify split membership, and report scope cardinalities and zero
   overlaps. The same checks must cover the fresh validation manifest (including
   historical validation tables) and the fresh test manifest. Validation and
   test fingerprints/source hashes must be frozen before training.
3. Preserve all pre-existing files listed in
   `results/LENGTH_MIXTURE_E14_PREFLIGHT.json`; enforce non-empty output guards
   for the new E14 run/result paths. Programmatically compute checkpoint,
   manifest, protocol, and source hashes rather than manually transcribing them.

## Implementation review evidence (2026-09-07)

I inspected the completed implementation and ran only read-only/bounded checks;
no training was run and no final E14 run directory was populated.

- `PYTHONPATH=. .venv/bin/pytest -q tests/test_length_mixture_e14.py`: **7
  passed**.
- A bounded Python smoke generated eight-record validation and test manifests,
  checked their disjointness, loaded all nine novelty scope names, and exercised
  the schedule helpers without writing artifacts.
- On the same seeded h2 batch and identical model state,
  `scripts/state_scale_train_e13.py:e13_forward` and
  `ReasoningModel.forward_with_readouts` produced exact step-4/final logits;
  the E13 and E14 h2 losses were bitwise equal and all parameter gradients were
  bitwise equal. This confirms the core objective implementation for h2.
- The inspected code has the registered schedule counts and sum-h, native
  budgets `4*h`, macro final validation score, six-cell gate
  (`h1@4`, `h2@4/8`, `h3@4/8/12`), seed-0 extension decision, selected-update
  guard, per-arm/latest checkpoint writes, actual base-h2 and mixture digest
  assertions, new validation fingerprint in the checkpoint, pre-training
  protocol file, and result overwrite guard.

## Remaining blockers before training clearance

1. **Mandatory novelty parsing is not fully fail-closed.**
   `novelty_scopes()` delegates E07 persisted/reconstructed and E08–E11
   handling to `scripts/state_scale_schedule_e12.py:_old_tables()`. That helper
   can skip malformed entries in persisted `eval_sets` records and only checks
   that the aggregate scope is nonempty. E14's `_records_from` also turns every
   `x["transitions"]` into a tuple without validating each record's type,
   nonempty canonical length, integer/range validity, or rejecting malformed
   records after a valid record. A required source with malformed records must
   abort, even if another file makes the aggregate scope nonempty. Add a strict
   per-source parser (including the E10 referenced manifest) or add a wrapper
   that validates every record and every required source independently.

2. **Frozen source provenance is incomplete.**
   `protocol.json` records hashes for the config, E14 script/tests, two new
   manifests, and baseline checkpoints, but not the mandatory novelty source
   files or `results/LENGTH_MIXTURE_E14_PREFLIGHT.json`. Since novelty sources
   and protected baseline artifacts are part of the pre-training protocol, their
   paths and SHA-256 values must be snapshotted in the protocol before the first
   arm starts. The later report should carry the same snapshot and fail if it
   differs.

3. **E14 checkpoint reload does not verify all arm provenance.**
   `load_mixture_checkpoint` checks the file hash, selected update, and presence
   in validation history, while `validate_checkpoint_payload` checks the basic
   tag/config/dynamics. It does not compare payload `initial_state_digest`,
   `data_stream_digest`/`input_target_sequence_digest`, `suite_fingerprint`,
   optimizer metadata, parameter count, or `teacher_forcing=false` against the
   arm report and protocol. A mutated payload with a correspondingly updated
   report could pass the local checks. The independent artifact review could
   catch this after training, but the guarded run should fail closed itself.

4. **Semantic test coverage is still insufficient for the registered gate.**
   The current seven tests do not positively exercise all six validation gate
   cells, deterministic selection/tie retention, label-target causal
   separation, negative novelty-source failures, actual-vs-independent schedule
   digest equality, or baseline checkpoint dynamics rejection. The bounded smoke
   and direct h2 gradient check provide evidence for those paths where tested,
   but focused tests are needed before treating the implementation gate as
   closed.

These are implementation/provenance blockers, not a conceptual feasibility
problem. The review remains **PENDING CODE** and does not authorize the guarded
training run until the four items above are repaired and retested.
