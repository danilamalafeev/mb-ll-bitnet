# E30 — oracle state, fresh last-operation diagnostic

2026-09-08. User approves next diagnostic after E29. Root owns design/reference/shared docs. One narrow Astra/low executor under documented escalation after repeated E27 Luna corrections; separate Astra/low reviewer. No training, new quantizer or new longer programs. All E28/E29 scientific decisions remain unchanged.

## Intervention and question
Can saved E27 W4/A32 and E24 float models correctly execute the LAST operation in isolation when supplied the TRUE numeric state after the first4 L5 instructions? Use all3seeds/model, original u16000 checkpoints, native8 internalsteps for this one fresh instruction. For each of the6 E28L5 programs compute trueprefixstate with exact DSL from original initialstate, not decoded model output. A fresh forward starts new input embeddings/cache/latent context; it is NOT an intervention only on hiddenstate. Compare its final predicted(x,y) with the saved uninterrupted E28 final output on the same program/initialstate/target.

Only last opcodes XOR and SWAP occur. Run each saved model once on all256lexicographic inputstates for XOR and once for SWAP. This yields12real forwards,3072unique model/primitive/state cases,3072readouts,24576internalsteps. Reuse these exact oracle state/opcode lookup results to map back onto all9216 E28L5cell-cases; no additional model forwards. The perprogram trueprefixstate maps are bijections of256states; verify rather than assume. Count lifted9216 comparisons separately, not as independent/model-executed cases.

## Prospective outputs and criteria

Freeze rootreference to E28report/selection/manifests, E29analysis/review, accepted6checkpoints/E27+E24manifests/reports/protocol. Validate allinputidentities/targets/strata and existing DSL before lookup. Use the same exact state/program keyed alignment, no ordering-only join.

Report freshprimitive correct counts /256 for eachmodel/seed/opcode; preregister diagnostic `fresh_exact_all` true iff ALL12cells256/256. This is local primitive sufficiency, NOT length5 success.

For eachmodel/seed/L5program save original initialstate, trueprefixstate, opcode, exacttarget, saved longtrace+finalcorrect, prefix4_all_correct, freshprediction/correct. Paired recovered(longwrong/freshcorrect), introduced(longcorrect/freshwrong), bothcorrect/bothwrong on ALLcases, and separately prefix4-correct versus prefix4-wrong strata and originalinitialstate train/validation/test. Fresh inputstate strata must not replace initialstate strata. Primary diagnostic `late_error_rescue_all` true iff across ALLmodel/seeds every saved final-error case with allfirst4readouts correct becomes freshcorrect, and no previously correct final in this prefix-correct subset becomes wrong. Denominator-zero rescued fraction=null, not100%; report evidence counts so vacuous groups are not called demonstrated repair. Aggregate subset has nonzero savederrors per frozenE28; verify.

Also report perprogram final freshcorrect/256 and former244threshold as a descriptive lifted metric only; never relabel E28 L5 gates or claim learned longer execution. No selecting only known errors, no seedselection/posthocthresholds. Preserve all selected cases, including longcorrect and earlier-diverged traces.

If fresh succeeds while long fails, support failure associated with the original long context/trajectory despite availability of the local numeric mapping; do NOT identify whether latent state, cache, reader, position/history or their interaction is causal. If fresh also fails, expose remaining local mapping errors and overlaps; do NOT infer long-context irrelevance. Oracle trueprefixstate supplies information not available to a free-running correction mechanism. No inference-time fix/generalization/packedkernel claim.

## Implementation, integrity and stop

Only new scripts/oracle_reset_e30.py and targeted tests/test_oracle_reset_e30.py, results/E30_* excluding root protocol/reference, fresh runs/e30_oracle_preflight/, runs/e30_oracle/ and uniquely named QA/review paths. Oldcode/runs immutable. Reuse accepted E28 strictloaders/evaluate_program or equivalent reviewedpurehelpers; do not monkeypatch oldconstants/evaluators. Verify model/optimizer/RNG/mode unchanged beforeafter including exception. Actual attempted/completed hook counters; numerical failure completes fullfixedscope, exception/nonfinite stops and preserves partial artifacts, no automatic retry. Refuse overwrite. Extend protected guard with prior artifacts plus E28/E29sources/science data and references; hashes before/after.

Independent code CLEAR before canonical preflight/scientific evaluation. Targeted puretests for oracle keyjoin, prefixtruevsdecoded, bijection, pairedmarginals, at-risk filtering/zero denominators, fullscope/identity/strata/malformeddata and overwrite/exception. One tiny LEGAL SEEN QA opcodeADD/two states per saved model is allowed separately with actual counters; do not inspect scientificXOR/SWAPfreshpredictions before gate. No training/suite expansion.

Final independent savedcheckpoint replay exactly12realforwards + rawstdlib recomputation of lifted9216comparisons/DSL/predicates/strata/provenance, separatecostaccounting. Stopfixedscope+review and HANDOFF/RESEARCH_LOG/ROADMAP update regardless numericaloutcome. No extra loops, states, operators, models, optimizations/cloud/E19/automation.
