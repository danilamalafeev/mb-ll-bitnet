# E32 independent final review

2026-09-08. **ACCEPT** — saved scientific artifacts, fixed budgets, provenance, predictions and reported predicates verified. This is acceptance of experiment integrity; the registered L5 restoration hypothesis failed for both precision arms. Reviewer e32_review Astra/low is independent of Luna/xhigh implementation/execution and root design.

## Result

All six models pass the seen prerequisite and E21 primary gate. Every model fails the all-six-program L5 final gate; both arm-wide primary and combined restoration are false. W4 passes L4 final/fulltrace across all seeds; float seed1 fails L4_5 at242/256. L5 fulltrace arm-wide gates also fail.

| Precision | h64 L5 errors seeds0/1/2 | h128 L5 errors seeds0/1/2 | Strict width benefit each seed |
|---|---|---|---|
| W4 |68 /21 /62 |159 /153 /165 |false /false /false |
| float |138 /6 /64 |146 /170 /109 |false /false /false |

L4 errors at128: W4 11/15/6, float7/20/8. Signed L5 W4-minus-float error gaps at128 are+13/-17/+56; h64 gaps are-70/+15/-2; interaction+83/-32/+58. All independently recomputed. These fixed opened functions, initialization recipes and training exposure do not establish general width harm, isolate hidden-state memory capacity, or identify a causal quantization effect. h128 changes335232parameter capacity, reader/head dimension and FFN expansion ratio while preservingFFN256/native8/opcode1/8. No reset was used.

## Independent evidence

- Strict reload and exact complete evaluation-dictionary replay of all six saved u16000 checkpoints: **498 forwards,72192 cases,223872 readouts,1790976 substeps;0 training updates**. Actual replay loop wall24.105991875s, including strict per-model reload and evaluation after initial preflight load. Each model/optimizer/RNG/mode digest is unchanged across evaluation. Checkpoint hashes and per-model replay artifacts are retained in `runs/e32_review_final/`.
- Independent stdlib raw audit: original DSL instruction targets, complete state/program coverage and ordering, prediction value/range/type, final/prefix/fulltrace flags, all strata, firstdivergence/recovery metrics, final/fulltrace paired precision and width counts, restored gates, strict benefit flags, signed gaps/interactions, and full498evaluation counters pass. `results/E32_review_raw.py` and `E32_review_raw_counts.json`. This script ran once,0forwards/updates; its wall clock was not instrumented and is not inferred.
- Additional independent stdlib seen/E21 comparison audit: **107520 case joins**, all12 model/comparator combinations; primary plus equivalent control, seen deltas, raw paired composition marginals and metric deltas pass. Exact per-model report copies, attempted/completed training/evaluation counters and all64 scheduled250-update progress entries verified. `E32_review_comparisons.py` / `E32_review_comparison_counts.json`; actual2.127788667s,0forwards/updates.
- Canonical preflight manifest equals regenerated manifest and saved run manifest. Initial tensors/RNG match the paired constructors; all final tensor shape/dtype/order, parameter count, optimizer state/membership/hyperparameters/step16000, checkpoint/manifest/source/initial/stream/RNG digests pass strict loader checks. Protected355 files and50 root references unchanged. Scientific model/runner/test hashes remain the exact CODE CLEAR versions.
- Scientific accounting:96000updates,6144000examples,12288000readouts,98304000training substeps, fixed evaluation498/72192/223872/1790976; all6 complete. No scientific retry or extra training. Sum of recorded training durations2153.838425416s is not command wall time; command wall/user/system were not captured and appropriately remain null.

## Review attempts and report QA

Final review had one successful scientific replay, one successful raw audit, one successful seen/E21 supplement,0reviewtraining,0failed replay attempts. Pre-scientific code QA is separate in E32_REVIEW_QA_ACCOUNTING.json: reviewer26updates/52examples/416training substeps plus6tiny evaluation forwards/96substeps, including one intentional failure probe. Executor QA attempts remain separate; the earlier two-branch QA counter undercount was preserved and disclosed, not overwritten.

Human report tables (L4/L5 final/fulltrace, gates, error gaps/interactions, training durations) match artifacts. Reviewer identified one prose ordering error: E21 six arrays were arm-major while labeled execution order. Exact seed-major correction was sent to executor; this affects presentation only and not accepted saved scientific arrays. Root/executor should use the corrected report and mark its pending review wording ACCEPT; no scientific rerun is warranted.

Stop condition reached: all fixed finals and independent audit complete. No width sweep, extra budget, reset, E19 or cloud is authorized by this review.
