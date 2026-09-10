# E15 register runner repair

2026-09-07. This note records implementation repairs only. It does not report
training, a gate outcome, a primary result, or a comparison result.

## Corrected evidence path

`scripts/register_interpreter_e15.py` now selects a checkpoint using validation
final joint accuracy macro-averaged by program length. Its tie-break is lower
final-position dual cross-entropy, also macro-averaged by program length, then
the earlier update. Validation rows used for that decision are embedded in both
the selected checkpoint and the completed arm metadata.

The seed-0 arms are each completed and durably recorded before the gate is
evaluated. A gate failure writes a terminal report and does not enter primary
or secondary inference. A pass permits seeds 1 and 2, then writes a durable
`training_complete` report before any primary, secondary, or control inference.
`--eval-only` is read-only recovery for that report or a completed report with
six frozen arms and a passed seed-0 gate; it cannot train, alter evidence, or
bypass a failure.

Paired outcomes are calculated per state from aligned final-joint booleans.
Thus equal aggregate accuracies can still yield both QAT wins and GRU wins.
Primary output retains every state, x/y target and prediction trace, prefix
joint flags, final joint count, full-trace count, and train/validation/test/all
state subsets. The report also includes identity, equivalent, symbolic,
order-sensitive, and cloned late-opcode controls.

## Frozen inputs and validation

The preflight manifest schema requires hashes for the E15 module, runner,
tests, authoritative protocol, DSL audit and audit script, and reused model,
quantization, and runtime modules. Checkpoints store the manifest/source
hashes, selected/latest tags, seed, update, parameter count, optimizer/config,
environment, initial-model digest, complete batch-stream digest, and the exact
batch-prefix digest at that checkpoint.

Focused E15 tests pass: semantics and split/schedule/parameter checks,
four-step replay and causal prefixes, checkpoint save/load guards, gate
positive/negative cases, paired equal-aggregate disagreement, final-CE macro
selection with unequal program counts, and CLI preflight/recovery surfaces.

## Remaining blocker

No scientific training or scientific inference has been run. A fresh preflight
must remain source-identical through independent review, which is the required
clearance before `--train-cleared` can be used.
