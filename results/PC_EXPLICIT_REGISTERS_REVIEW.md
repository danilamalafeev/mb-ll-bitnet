# Independent explicit-registers review — ACCEPT

2026-09-09. Reviewer pc_review (Astra/low). The registered saved-data experiment is **ACCEPTED**. No reviewer model forwards, checkpoint deserializations, QA reruns, baseline reruns or numerical replay were performed.

Every ADD, XOR and SWAP table is exactly correct on all256 register pairs for all six B endpoints (float/W4, seeds0/1/2): **0 errors in4608 primitive predictions**. Independently composing each table's own predicted outputs reproduces the correct full DSL trace for **all105984 program/state trajectories and1990656 positions**, including all length32 programs and all inherited initial-state strata.

## Independent verification

Fresh code-project artifacts under `runs/pc_explicit_registers_v1/`:

- `audit_science.py`: standard-library oracle and independent composition/metric implementation; no project imports; bounded endpoint-wise baseline reads;5.625 seconds.
- `science_raw_audit.json`: all per-program metrics, per-endpoint/suite/length/stratum aggregates, heldout64 aggregates, primitive errors, hashes and accounting.
- `science_summary.json`: compact per-endpoint/suite and pooled length/stratum counts for reporting.

All saved per-program train192/validation32/test32 final/full-trace counts, paired outcomes, first-error/recovery histograms and carried-state/local-step cells exactly equal independent recomputation. Heldout64 is recomputed as validation32+test32 and verified against inherited accepted state lists. Every baseline target trace matches the independent ADD modulo16/XOR/SWAP oracle; endpoint/checkpoint/program/state joins are exact. Original baseline correctness flags were not used as oracles.

The immutable science manifest still has SHA256 `26670c8010349ab983b288c505b69a96452f47a8f75767b1547a05b35cc72eba`. Frozen source, checkpoint, program, baseline, protocol, accepted-manifest/lineage and QA hashes match current bytes. Supplemental accepted model-source inventory and retained QA helper/loader bindings match. Saved model/RNG/mode identities are unchanged on all endpoints. Report endpoint metadata matches the endpoint artifacts exactly.

## Results

| Suite, six B endpoints pooled | Cases | Baseline final correct | Explicit final correct | Baseline full correct | Explicit full correct |
|---|---:|---:|---:|---:|---:|
|Padding|69120|21506|69120|19764|69120|
|Compositions|36864|15524|36864|14865|36864|
|Padding, heldout64 initial states|17280|5435|17280|4980|17280|
|Compositions, heldout64 initial states|9216|3831|9216|3672|9216|

Every endpoint improves over its saved baseline in each complete suite, for both final and full-trace outcomes. Padding recovers49356 baseline full-trace errors; compositions recover21999. **No introduced errors and no both-wrong cases.** Existing correct baseline cases remain ties; in particular all13824 L4 padding cases were already fully correct and remain fully correct. These pooled descriptions supplement, rather than replace, saved endpoint/length/stratum results.

AtL32 the full-trace comparison is padding51/13824→13824/13824 and compositions12/9216→9216/9216. Every carried input and every local primitive step in explicit composition remains correct; all1990656 positions fall in the true-input/local-correct/output-correct cell. There are no first errors or later recoveries within the explicit traces themselves. “Recovered” in the paired comparison refers to improvement over continuous-baseline cases, not a correction after an explicit-chain arithmetic error.

## Accounting and limits

Science records18 attempted/completed forwards,4608 one-op cases/readouts,36864 native steps,6 endpoint load requests,28 underlying checkpoint deserializations and0updates, with no failures. Accounting agrees across report, terminal status and progress. The separate accepted QA cost remains2 forwards,4 cases,32 native steps and2 deserializations; combined new-experiment totals are20 forwards,4612 cases,36896 native steps and30 deserializations. Derived long trajectories are lookup work, not additional neural forwards.

Recorded science time is285.968 seconds:252.453 loading,0.485 table construction and32.453 lookup/composition including metric work. The detached job took approximately329 seconds including preparation. These are different time boundaries; table lookup is not an uncached sequential-neural runtime benchmark. Background status confirms exit0 and one queued completion wake. `progress.json` retains its running label; terminal `status.json` and `report.json` correctly say complete and all counters agree. This cosmetic nonterminal label does not invalidate the result and needs no experiment rerun.

Because all three measured primitive maps equal the DSL over the entire finite state domain, induction establishes exact composition for any finite opcode string **within this frozen table-defined system**. No longer program was run neurally, and this does not prove numerical equivalence under arbitrary online batch shapes, arbitrary-language reasoning, or continuous latent-state generalization. Fresh per-operation execution changes both hidden state and initial-register KV; this result does not isolate hidden-state drift as the sole causal mechanism. The reused, previously opened program pool makes this an adaptive diagnostic, not a fresh heldout confirmation. Initial-state heldout labels are not claims about unseen intermediate states.

The old continuous-science protocol deviations remain recorded in its own review; they are not retroactively repaired by this experiment. This prospective experiment passed its pure/code/QA/freeze gates and final independent saved-data audit. Stop at the registered scope; no old replay or follow-on work was started.
