# Independent PC CUDA length review — 2026-09-09

Решение пользователя 2026-09-09: повторные 12 model forwards отменены и исключены из текущего объёма работ. Независимое ревью сохранённых данных завершено: all-case audit PASS. Replay не выполнялся и не остаётся ожидающим шагом. Новые прогоны ради закрытия этого ревью не запускать. Исторические ограничения протокола сохранены.


Reviewer: pc_review (Astra/low). Scope: read-only audit of every saved science case, independently derived DSL maps and metrics; no training, checkpoint deserialization or model forward in this audit. Original science artifacts were preserved.

**Verdict: saved scientific data pass the all-case consistency and semantic-selection audit. Full protocol compliance is not established and cannot be retroactively claimed. Narrow numerical replay remains pending.** This is not a full model replay or an independent certification of historical runtime costs.

## Evidence and scope

Fresh artifacts live at `C:/Users/я/LoopedBitNet_AI2_inference/project/runs/pc_inference_v1/review_v1/`:

- `audit_raw.py`: standard-library implementation with no project imports; explicit ADD modulo16, XOR, SWAP oracle, hand trace/pair fixtures; 11.375 seconds.
- `raw_audit.json`: complete per-program/endpoint/length/suite/initial-state-stratum final and full-trace counts; independent A/B recovery/introduction/ties; anchor/member final and full-trace pairs; corrected conditional final-O counts with eligible/excluded denominators; first-error positions; checkpoint and raw-file SHA256 inventory.
- `audit_combined.py` / `combined_identity.json`: bounded streaming comparison, one endpoint at a time; all12 records in the 829 MB combined report exactly equal endpoint files; 21.453 seconds.
- `derived_summary.json`: descriptive pooled counts and first-error distributions, retained alongside the unpooled audit data. Pooled results do not replace seed/precision/stratum results.

All828 unique endpoint/program rows contain all256 states exactly once. All211968 target traces match the independent DSL; every predicted register is an integer0–15, with the correct trace length. Stored correctness flags and row final/full/prefix/register summaries match independent recomputation. Totals are3981312 instruction readouts and31850496 native steps (8 per instruction per case). These are saved-output workload counts, not recovered attempted-runtime counters.

All12 checkpoint bytes match each endpoint's stored hash. This verifies current byte identity against the saved records, not a missing prospective source freeze. The original manifest omits a complete source/checkpoint/backend-setting inventory. Historical strict-loader/provenance assumptions and Windows loader substitution require separate code review; raw correctness does not validate them by itself.

## Semantic identity and selection

The accepted E36 manifest provides125 trained programs:32 allowed short programs and93 length4–6 pool programs. Independent full-domain maps of every nonempty prefix give94 functions. The saved primary allowed semantic-new length7–10 pool gives23 functions. Independent deterministic candidate-stream reconstruction (`random.Random(20260909 + length)`, ADD/XOR/SWAP order) selects exactly the saved first six qualifying maps at each length. Draw/distinct counts are28/70/166/257 at lengths12/16/24/32, within both caps. All24 selected composition maps are distinct across lengths as well as within each length, and excluded-set novelty holds. This is finite-set semantic novelty, not minimal-length proof.

All45 padding strings match the prescribed nine families and five lengths; final functions and true pre-O states equal their own L4 anchors on all256 states. Nine families represent **eight distinct functions**, and **every anchor function is present in the94 trained-prefix functions**. Thus padding and compositions differ in training-function membership; they are not interchangeable length controls.

Inherited state membership independently matches the accepted192 training and64 heldout lists. The audit preserves validation32/test32 separately and their heldout64 union. These are initial-state strata, not unseen intermediate-state guarantees.

## Corrected metrics and scientific interpretation

The original `conditional_final_O` requires correctness of the entire earlier trace. Contract semantics require only that both paired pre-O register readouts are correct. The review recomputes this from the second-last readout alone and compares final-O correctness among eligible pairs.

| B padding length | Eligible | Excluded | Long final-O errors among eligible |
|---|---:|---:|---:|
|12|8163|5661|2618|
|16|1086|12738|737|
|24|300|13524|214|
|32|192|13632|129|

These descriptive pooled denominators are13824 pairs per length across six B endpoints. The long-length eligible subset is very small; conditioning selects survivors and does not isolate a hidden-state causal mechanism. All B L4 anchor outputs are correct, so these errors are introduced relative to the paired short anchor.

Independent full-trace counts reproduce the previous descriptive table: B compositions8641/9216 atL12,5827/9216 atL16,385/9216 atL24,12/9216 atL32; B padding5509/13824,305/13824,75/13824,51/13824 respectively. AtL32 padding A has53 full-correct versus B51, and A306 final-correct versus B184. AtL24 padding A349 final-correct versus B259. Do not infer improvement at every length or for both metrics from suite-total benefit.

Explicit zero-error A ties: atL4 padding, float seed0 and W4 seed2 A each have2304/2304 full-correct, tying their B counterparts. These are ties, not B recoveries.

B padding L32 first errors concentrate at instruction9–14 (12312/13824 cases), with earliest error at7; this suggests failures often precede the final instruction. B composition L32 first-error counts peak at16 (1155),17 (1116),18 (1011). First-error distributions are descriptions, not evidence for a particular internal mechanism. Composition sets differ across lengths, so cross-length performance is not a paired-program degradation experiment.

## Protocol deviations and accounting boundary

The results document accurately acknowledges missing prospective tiny QA and CODE CLEAR, incomplete source/backend freeze, no automatic migration-report gate, incorrect conditional metric, constant terminal counters, and shell-session waiting instead of a detached completion hook. CPU/CUDA migration was interleaved per endpoint; seed0 used a different L7 selection category, and selection used opcode tuples rather than the contract's stored-ID rule. These historical deviations remain deviations even if replay passes.

The raw archive proves828 saved scientific batches; it cannot establish all failed attempts, total deserializations, or exact historical wall/runtime accounting. Zero training is consistent with the inspected inference runner, but there is no reconstructed global process ledger. No science rerun is warranted solely to improve bookkeeping. The required prospective narrow replay must be counted separately:12 forwards,3072 cases,786432 native steps,0updates, first padding family's fixedL32 (`padding_ADDADD_ADD_k14`) on all12 endpoints. It must compare every decoded trace exactly, freeze current source/inputs/settings, retain attempted/completed/failure/load accounting, and use the reviewed detached launch. Replay code/launch review and replay outcome will be recorded separately when available.

Stop: raw audit completed. No new length, seed, training, architecture, precision experiment or scientific sweep was started or authorized by this review.
