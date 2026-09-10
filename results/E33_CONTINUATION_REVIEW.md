# E33 independent final review

2026-09-08. Reviewer e33_review, Astra/low. **ACCEPT**: the fixed seed0 continuation and saved artifacts are valid. This is acceptance of the evidence, not a successful restoration result.

Both scientific u32000 checkpoints were loaded through the strict E33 loader. Absolute optimizer steps32000, complete model/optimizer/RNG/mode identity, parent/source/config/stream provenance and protected references were checked. Replayed the unchanged E32 final scope exactly once per arm:166 forwards,24064 cases,74624 readouts,596992 native internal steps. Both complete evaluation dictionaries, including every saved prediction, exactly equal the canonical reports. Model, optimizer, Torch RNG and training mode remained unchanged. No training updates or additional QA were performed for review.

The separate stdlib audit recomputed DSL targets and prediction-derived final/fulltrace/prefix/divergence/recovery metrics, original state coverage and strata, own-parent joins, per-stratum L4/L5 pairs, seen/E21 comparisons, W4/float pairs, gates, costs, progress schedule and file hashes. All passed. Canonical and per-arm reports agree. Scientific paid work is32000 added updates across both arms,2048000 examples,4096000 readouts,32768000 native steps; prior parent work is separate.

| Seed0 arm | Parent L5 errors | u32000 L5 errors | Strict benefit | L4 errors | L5 restoration |
|---|---:|---:|---|---:|---|
| Float128 |146|166|False|5|Fail|
| W4 128 |159|133|True|7|Fail|

Both seen, E21-primary and L4-final prerequisites pass. Both L4-fulltrace gates pass; both L5-final, L5-fulltrace and combined restoration gates fail. Paired training benefit is False. Signed W4-minus-float L5 error gap is-33, versus parent+13. Float recovered83 parent errors and introduced103; W4 recovered122 and introduced96. The intervention improved this W4 seed but worsened its float counterpart; it did not establish paired benefit or restoration.

Replay artifacts: `runs/e33_review_final/{float128_seed0,w4128_seed0}.json` and `review.json`. Independent raw counts: `results/E33_review_raw_counts.json`. Review scripts and logs are `results/E33_review_replay.py`, `results/E33_review_raw.py`, and corresponding logs. Replay command measured real78.96/user75.31/sys2.62 seconds. Successful raw audit measured real1.38/user0.88/sys0.31 seconds and used0 forwards/updates.

One initial raw-audit attempt stopped at a reviewer-script field-name mismatch (`name` versus `program` in accepted precision-pair rows). It used0 forwards/updates; its log is preserved as `E33_review_raw_attempt1.log` (real0.58/user0.49/sys0.06 seconds). Only that independent audit script was corrected, then raw auditing completed. No canonical artifact or replay was retried. Earlier interrupted implementation QA has unknown cost as already disclosed in E33_QA_ACCOUNTING.json; do not infer zero from absent records.

Limitations remain the preregistered ones: one adaptively selected continuation setting on opened data, seed0 only, no equal32000-budget h64 comparison, no significance/noninferiority or quality-ceiling claim. No additional seed, longer continuation or architecture change is authorized by this outcome. Stop at this accepted fixed result and update the shared handoff.
