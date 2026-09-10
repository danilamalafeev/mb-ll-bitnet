# E18 first-attempt technical failure and bounded repair

2026-09-07; Astra/low reviewer and narrowed repair owner.

The first scientific attempt failed with NameError: PROGRESS_INTERVAL was
referenced in train_arm but missing from the runner import list. It failed
AFTER the first steps4 optimizer.step and BEFORE any progress entry/checkpoint.
There is one consumed update,64 examples and256 recurrent substeps; steps8 did
not start. There are no final scientific results. Preserve the failed run and
original preflight; never resume this partially consumed model state.

The previous independent CLEAR was wrong about executable runner readiness.
The reviewer inspected training code but focused tests performed updates
manually; they never executed train_arm. Even106 passing tests did not exercise
this path. This is a concrete coverage failure, not a scientific null result.
The earlier CLEAR and HOLD remain as historical records and do not clear the
modified runner on their original hashes.

## Repair and meaningful verification

`E18_STEP_BUDGET_TECHNICAL_REPAIR.patch` adds only the existing constant
PROGRESS_INTERVAL to the runner's module import list; its value is250 as
registered. No recurrence, optimizer, data, loss or scientific criterion changes.
Root retained the exact old source tree under `E18_FIRST_ATTEMPT_SOURCE/` and
accounted the failure in `E18_FIRST_ATTEMPT_ACCOUNTING.json`, then authorized
applying this one-line repair. Repaired runner SHA256:
`6c43e4e7266057232fc2f5d4e687578b4983e2b257625c8d52f0ade561be6b27`.

`E18_STEP_BUDGET_TRAIN_ARM_SMOKE.py` executes the ACTUAL train_arm path. It loads
the preserved original source, reproduces its NameError, applies the exact
one-line repair in a temporary module, then completes one tiny update in EACH
budget including progress, checkpoint and completion-record writing. It asserts
actual parameter changes and native budgets, and the live repaired source equals
the tested patch. Each tiny batch is explicitly one legal length2 example.
UPDATES=1 is a QA monkeypatch only. Synthetic temporary checkpoints retain the
runner's u2000 label; they are never scientific evidence and are deleted.

The smoke passed before application and again against the preserved original
snapshot after application. Each invocation costs3 tiny updates/3 examples/
32 recurrent substeps including its original-failure reproduction; total QA
cost across the two invocations is6 updates/6 examples/64 substeps. All138
protected hashes passed. No scientific retry/training was performed by this
agent. Original module/tests/protocol, failed run and preflight remain intact.

## Prospective scope recommendation

The original protocol explicitly prohibited retry. A new prospective technical
amendment must supersede that clause narrowly BEFORE another update; silently
retrying under the old protocol would be invalid. Under the user's continuing
authorization to complete the experiment, a single technical restart is
scientifically defensible: the failure produced no model evaluation or outcome
used to select a new scientific intervention. Keep the first attempt disclosed.

Use fresh paired initialization and fresh optimizers for BOTH arms, the SAME
frozen data/targets/2000 updates/native budgets, unchanged reporting and stop
criteria. Freeze repaired sources in a NEW preflight and use NEW run paths
`runs/e18_step_budget_repaired_preflight/` and
`runs/e18_step_budget_repaired/`. Do not reuse the consumed update, overwrite
failed artifacts, compensate the budget, add seeds or inspect reserved sets.
Only this mechanical repair/restart exception is justified; independently
review the resulting pair and stop. Root owns the amendment and run decision.
