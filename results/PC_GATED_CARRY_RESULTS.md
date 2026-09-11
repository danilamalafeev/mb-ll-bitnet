# Gated carry paired pilot — saved result

## Outcome

The fixed two-arm pilot completed successfully with no runtime failures. The
independent saved-artifact audit accepted the trace integrity and reproduced
the runner's paired counts without loading a model or checkpoint.

The primary pilot predicate is **not met**. After 1,000 updates per arm, the
unchanged writer arm A is perfect on the evaluated pool (38,400/38,400 full
traces). The learned gated carry arm B reaches 38,310/38,400 (90 full-trace
errors): 11,466/11,520 on old padding, 6,144/6,144 on old compositions, and
20,700/20,736 on the new identity controls. Thus B has 54 regressions on old
padding and 36 regressions on new controls, with no repairs relative to A.

The gated arm improved substantially from its own initialization: it rose from
10,423/11,520 padding, 6,133/6,144 compositions, and 19,874/20,736 controls
to the final counts above. The unhooked arm rose from 11,484/11,520 padding,
6,144/6,144 compositions, and 20,712/20,736 controls to a perfect final
evaluation. The final comparison therefore favors the ordinary carry path
under this joint training package and fixed seed.

## Scope and accounting

- Training: 2,000 updates and forwards total, 128,000 cases, 447,488 training positions.
- Evaluation: 531 computed forwards, 135,936 cases, 2,792,448 positions.
- Saved rows: 600 (69 reused accepted baseline rows plus 531 computed rows).
- Total: 2,531 forwards, 263,936 cases, 3,239,936 positions, 25,919,488 native forward steps.
- Checkpoints: local 0, 250, 500, 750 and 1,000 for each arm; all 10 ledger entries are present and hash-matched.
- Deserializations: 6 endpoint loads; no checkpoint reloads during science.

All attempted and completed counters match, with zero recorded failures. Gate
summaries are finite and in [0, 1]; training progress has 1,000 finite losses
per arm.

## Independent verification

The stdlib-only audit is [audit_pc_gated_carry.py](../scripts/audit_pc_gated_carry.py)
and its saved output is
[science_saved_audit.json](../runs/pc_gated_carry_v1/science_saved_audit.json).
It reconstructs E15 DSL targets and hash-ranked strata, validates every state
and target trace, checks all 600 row joins, compares every paired category with
the runner report, and verifies the computed case/position budget. Result:
`ACCEPT_TRACES`, `model_calls=0`, `checkpoint_loads=0`,
`runner_report_match=true`.

The source/freeze and accepted QA bindings remain in
`runs/pc_gated_carry_v1/science/{source_binding.json,input_freeze.json}` and
`runs/pc_gated_carry_v1/qa_accept.json`. The run used the cleared source gate;
the accepted pure and QA runtime files were unchanged.

## Interpretation and limits

This is a negative result for the registered success criterion, not evidence
that a gated carry mechanism is impossible. The gate was trained jointly with
the original adapter from one accepted endpoint and one fixed stream/seed, so
the experiment does not isolate learned gate values from the rest of the
adaptation. The finite pointer-chasing controls do not establish broad
reasoning or language generalization. No threshold search, extra seed, rescue
tuning, replay, or new experiment was run.

The fixed pilot is complete. Preserve the artifacts and stop here unless a new
experiment is explicitly authorized with a new protocol and budget.
