# E35/E36 independent review

Reviewer: length_wave_review, Astra/low. 2026-09-08.

## Phase 1 — protocol

Protocol decisions registered before model science in `results/E35_E36_PROTOCOL.md`. Exact A8000/B4572 instruction match verified arithmetically:16002 batch-instructions each. B8000 has exactly4000 short/4000 long batches. A explicitly repeats ancestral short data, so unique-composition/example diversity remains part of the intervention. Directional primary comparisons are per precision at both compute slices; training attainment is sampled and separate from heldout transfer. No extrapolation to an architecture limit or minimum semantic length from incomplete search.

Gate: PENDING manifest feasibility confirmation and targeted independent code review. No reviewer model loads, forwards or training performed in this phase. This is not CODE CLEAR or final artifact acceptance.

## Phase 2 — targeted code review in progress

Initial review found and returned three blockers before scientific execution: B4572 was a separate repeated training run instead of a snapshot of B8000; manifest regeneration included nondeterministic elapsed seconds; E35 paired identities incorrectly compared complete traces of different lengths. E35 final/pre-O identity comparison and standalone parent-only runner were corrected. Repeating E35 on new E36 checkpoints was removed as unregistered scope.

Reviewer pure DSL fixture (36programs ×256states) confirms pre-O/final identity and all27pair-family denominators, using perfect synthetic decoded traces;0model loads/forwards/updates. E35 inference uses accepted E33 loader and DSL evaluate_program, fixed all-state rows, forward hooks, and unchanged model/RNG/mode check. E35 source/manifest freeze remains required for its independent CODE CLEAR; E36 snapshot/reload review remains pending.

## E35-only CODE CLEAR

**CODE CLEAR: E35 only**, using the exact preserved source `runs/e35_e36_preflight/source_snapshot/scripts/length_wave_e35_e36.py` and manifest `runs/e35_e36_preflight/manifest.json` (SHA256 `2e18ffe8664e56c1bf5384f28e6e74d4abea17a9eb40f253e8ac5cabe8bc4177`). Reviewer verified every manifest source hash against its preserved source snapshot and both E33 parent checkpoint file hashes against the manifest. Fixed36programs per precision,9216cases/55296readouts/442368native steps per precision;27pair families,256states each. Source matches reviewed repaired E35 pair logic and accepted strict E33 loader path. E35 execution must import this snapshot and explicitly pass the repository root, reading the already-frozen manifest directly rather than re-freezing it.

The original preflight used an expensive inherited E33 manifest loader that internally deserializes parents. Its earlier exact preparation-load count is unavailable and must not be reported as zero; no evidence of scientific forwards or updates in that preparation. Root's later nonempty-directory refusal occurred before maker/model loading, hence0loads. This preparation defect and pending E36 parent-cost/schema fixes do not affect the E35-only scientific computation. No E36 training clearance is granted here. Final saved E35 artifact review is still pending.

## E35 final saved-artifact gate — ACCEPT

**ACCEPT E35**, independent raw audit plus one exact prediction replay complete. `results/e35_e36_review_audit.py` independently reconstructs DSL targets, all per-position correctness flags, row totals and every paired count from18432saved cases; results in `results/E35_REVIEW_RAW.json`. All36programs/256states per precision and both parent hashes match the frozen manifest. Replay used the preserved E35 source with a directly SHA-verified accepted parent manifest and the unchanged strict E33 checkpoint loader. All72replayed program outputs agree exactly with saved rows (`results/E35_REVIEW_REPLAY.json`):18432cases,884736internal steps,2loader calls/4checkpoint deserializations,0updates,44.65seconds. This is reviewer evaluation cost, separate from the scientific run.

| Precision | L3 final/trace | L5 final/trace | L7 final/trace | L9 final/trace |
|---|---:|---:|---:|---:|
| float |2304/2304|2270/2270|1588/1484|660/368|
| W4 |2302/2302|2249/2247|1012/930|258/77|

Every numerator has2304cases as denominator; each cell shows final-correct / full-trace-correct, not a ratio. Exact identity insertions preserve final function and pre-O true state, yet both models degrade. Even where both immediately preceding readouts are correct, short-correct/long-wrong O outcomes are float34/2304,453/1941,407/798 and W4 52/2301,803/1740,200/309 for k1/2/3; reverse outcomes0in these supplemental strata. Thus a new final function is not necessary for late execution errors on these fixed histories. Additional SWAP operations alter history composition and compute together; this does not isolate h, KV, reader or their interactions, prove a universal length threshold, or compare precision statistically. E36 remains a separate pending gate.

## E36 v2 freeze and pre-science scope clarification

Frozen `runs/e35_e36_preflight_v2/manifest.json` SHA256 `4adf2dd1bb1036b94bcbf7a0246d9410596a737c9d20e5ff8865632fab5988aa`: reviewer verified current source hashes and both parent hashes. Exact pools/schedules/probes equal the original prepared data; only preparation/loader implementation changed. Independently reconstructed full256-state semantic maps for all training prefixes and selected evaluation programs:94exposed functions, zero overlap for every semantic-new selected program. A contains all32legal L1–3 strings, matching the ancestral syntax scope. Primary allowed L7–10 contains6programs/length =24programs/6144cases. Long training pool sizes29/32/32 for L4/5/6.

Before E36 science, coordinator explicitly accepted two reporting clarifications: `train_syntax` is exposure in the UNION of ancestor/A/B, so it must not be called seen-training syntax for A at L4–6; own-branch sampled training probes remain separate. The separate historical six-program L5 regression evaluation is omitted in this bounded wave, a deviation from the protocol text. It must be disclosed in the final report; the primary novel pool and all budget/attainment endpoints are unchanged. Frozen protocol bytes were preserved rather than silently edited after manifest freeze. E36 CODE CLEAR awaits the single bounded snapshot QA.

## E36 CODE CLEAR

**CODE CLEAR for the four registered E36 continuations**, using v2 manifest/source hashes above. The accepted-parent manifest is now read directly after SHA verification; actual parent restoration still uses the strict E33 checkpoint loader. E36 uses cumulative E33 lineage fields, validates parent hashes/digests, manifest/source/RNG/model inventory and schedule-derived costs. A runs8000updates; B runs8000once and saves a4572snapshot in the same trajectory, with snapshot evaluations only after training ends. No duplicate B-prefix training is present.

Saved bounded QA `runs/e35_e36_qa_v2/report.json` completes both precisions: A reload-next-update equality and B snapshot-next-update equality for model, optimizer and RNG; matched checkpoint32001 and final32002. QA manifest equals scientific v2 byte-content, all current source hashes still match. Exit0,378.73seconds (`results/E36_QA_TIME.txt`). Static execution-path accounting includes all extra equality updates:7updates per precision (A2 + two A comparison updates + B2 + one B replay update),14total;896training examples,1280readout positions,10240internal steps. Primitive QA evaluation adds2forwards/4cases/4positions/32steps. The report's training_cost field alone covers only initial A2updates and must not be mistaken for all QA. Loader calls total10 (4accepted-parent +6new-checkpoint loads),20checkpoint deserializations from these paths; no training outside this bounded QA is attributed here. Reviewer did not repeat model QA. Final scientific artifact acceptance remains pending.

## Final E36 saved-artifact gate — ACCEPT

**ACCEPT E36.** Independent `results/e36_review_audit.py` reconstructed all341376saved evaluation cases and DSL traces across A8000/B8000/B4572 for both precisions, verified exact manifest program/state coverage, checkpoint file hashes, every row's final/full-trace/per-position counters, snapshot identity and schedule-derived training/evaluation costs. Results: `results/E36_REVIEW_float_raw.json`, `results/E36_REVIEW_w4_raw.json`. Additional saved optimizer inspection confirms all32optimizer entries per checkpoint have absolute step40000 for A/B and36572 for B_match (`results/E36_REVIEW_OPTIMIZER_STEPS.json`),6extra checkpoint deserializations,0forwards.

One strict replay of each of the6saved checkpoints passes exact equality of all evaluation/probe rows. Artifacts `results/E36_REVIEW_float_replay.json` and `results/E36_REVIEW_w4_replay.json`;3loader calls/6deserializations per precision,170.90/172.80seconds respectively, actual4CPU threads as in science. The intended2thread process setting had been overridden by the unchanged model factory; both scientific workers and replays actually used4threads, preserving the parent setting. No restarts or model-budget changes followed this observation. Replay cost total1374forwards,341376cases,2096256readout positions,16770048internal steps,0updates, recorded separately from science. No further model checks are needed.

| Precision | A8000 errors | B4572 errors, instruction-matched | B8000 errors, update-matched |
|---|---:|---:|---:|
| float |4006/6144|14/6144|12/6144|
| W4 |4785/6144|30/6144|14/6144|

These are full-trace errors on the registered allowed semantic-heldout L7–10 primary pool. Both precisions meet the directional criterion at both budget comparisons. The24program strings represent23distinct full-domain functions (one cross-length alias); all are disjoint from the94training-prefix functions. They must not be described as24independent semantic classes.

By length L7/L8/L9/L10, with1536cases per length: float A errors435/1044/1099/1428, B4572 0/0/4/10, B8000 0/0/4/8; W4 A698/1226/1354/1507, B4572 0/0/5/25, B8000 0/0/1/13. Improvement therefore includes L10 itself rather than only an aggregate dominated by shorter lengths.

All own-branch training probes have0full-trace errors: A2880/2880correct and B/B_match6336/6336correct, including3456/3456L4–6 probes at both B checkpoints. This exceeds the registered99percent sampled-attainment threshold; it is not an exhaustive assertion about every training example. Each precision trained exactly8000A +8000B added updates; the4572snapshot adds no separate training. Across four continuations this is32000added updates,2048000examples,5632000readout positions and45056000internal steps.

The pilot supports the benefit of the package of longer training trajectories/compositions over repeating short training, including at equal processed instructions. It does not separate length from composition/semantic/example diversity, establish general reasoning or extrapolation beyond L10, or provide a multi-seed precision comparison. E35 shows failure on unchanged functions is possible in the original checkpoints; E36 shows this bounded transfer deficit is strongly reducible through the registered training intervention. Historical separate six-L5 regression omission and union-exposure labeling remain the disclosed pre-science deviations above. Fixed-wave stop is met: E35 ACCEPT and E36 ACCEPT, no further training or review forwards requested.
