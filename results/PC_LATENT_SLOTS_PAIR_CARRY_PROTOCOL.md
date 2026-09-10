# Latent padding pair-carry intervention — prospective protocol v1

2026-09-10. Owner: root task 01a08b1b-e103-77b3-b360-5734cf71e230. User authorized continuing research. New bounded inference-only study; accepted science/diagnostics remain immutable. No training, new seed, architecture sweep or replay of all69 programs.

## Question and design

Does accumulation through repeated identity SWAP pairs causally contribute to the saved endpoint's long-padding errors? Compare the same frozen latent local2000 endpoint under (A) sham/no-op hooks and (B) pair-carry restoration. This is an external program-aware intervention, not a learned memory solution, not proof of a unique faulty writer, and not a generalization claim.

Use exactly the accepted18 focus programs at L24/32 and all256 initial states each. Programs have two useful prefix instructions, an ODD21/29-SWAP segment, and a final useful instruction. Pair padding positions(3,4),(5,6),...,(21,22) or(29,30), one-based; preserve the last unpaired SWAP at23/31 and final useful instruction at24/32. Do not freeze individual SWAP operations or infer identity from target labels.

Before each registered pair, clone its input z. Execute both ordinary model instructions and emit their unchanged logits. After the second instruction has computed its ordinary writer output, replace only the carried z with that pair-entry clone in intervention B. In A retain ordinary writer output (clone/no-op). Never rewrite the current instruction's logits, supply correct registers/targets, skip model steps, carry previous h/KV, or change weights. A restored pair may still have an incorrect readout inside it. Both arms use the accepted latent forward and fresh-cache code unchanged, preferably scoped reader/writer hooks removed on exit/error. No intervention outside the registered pairs.

## Evidence and acceptance

The sham arm must reproduce the saved accepted decoded focus traces exactly; mismatch blocks scientific interpretation, no retry. Compare B against A per state/program/length/stratum for full-trace and final correctness, repairs/regressions, first error, and wrong positions in padding versus useful suffix. Recompute target traces with the existing DSL outside forward. Report original36 failures separately, and all4572 originally correct controls; do not cherry-pick only repaired cases. A clear positive pilot is all36 full-trace repairs and zero new full-trace regressions; otherwise report partial/null/harmful result without tuning. Use paired raw counts as primary evidence, no post-hoc threshold selection. The36 failures repeat across prefixes/lengths/suffixes and map to only two semantic prefix-end states(4,14)/(14,4); do not treat them as36 independent discoveries or compute an independence-based significance claim.

Optionally record compact per-pair signed-vector displacement L2(raw writer output minus pair-entry z), not all hidden tensors. This is descriptive; restoration effect establishes only the consequence of this intervention at this opened endpoint/pool. It cannot distinguish writer from reader/core/readout interactions or demonstrate a stable cycle.

## Gates and fixed budgets

First deliverable: new intervention helper plus independent toy fixtures only. Required fixtures: exact pair schedule and rejection of malformed/non-SWAP/overlapping/out-of-range pairs; odd tail preserved; ordinary intra-pair readouts unchanged; only future slot input restored; no-op equivalence; no target input; handles removed on success/failure; hand-calculated repair/regression/tie and accounting checks. Independent oracle must deliberately make one SWAP nonidentity and two SWAPs identity, while adding a detectable write perturbation, to catch freezing a single SWAP or replacing readout labels.

Then one disposable inference QA on first2 states, program ADD ADD SWAP SWAP ADD: accepted unhooked, sham, intervention =3 program forwards,6cases,30positions,240native steps,0updates. Check exact unhooked/sham logits; intervention logits equal through pair-end position4 and restored input consumed at position5; finite outputs, unchanged model/adapter/optimizer/RNG/mode. One strict accepted endpoint load (expected3 underlying deserializations), no training/snapshot QA needed because no update path changes. QA uses a fresh output and never changes science inputs.

After pure tests and QA acceptance/CODE CLEAR, freeze input/source/protocol/QA/checkpoint/saved-reference hashes once before science. Science:2arms x18programs =36forwards,9216cases,258048positions,2064384forward native steps,0backwards/optimizer updates. Exactly one strict endpoint load (expected3deserializations) in science; QA cost separately retained. Count actual attempts/completions and load calls, preserve partial costs, fail fast. No extra real-model checks beyond these budgets. Reuse accepted manifest/settings/strict loader; do not rebuild original training preflight.

## Outputs and stop

New code: scripts/pc_latent_slots_pair_carry.py and focused tests; later new runtime file. New runs/pc_latent_slots_pair_carry_v1/{qa,science}; background jobs separately named. Existing source/checkpoints/results must not change. Root owns shared docs/handoff/state. Use existing run_and_wake.py with this task ID for potentially long QA/science; one completion wake and no model polling. On completion audit saved traces/accounting only, write results/PC_LATENT_SLOTS_PAIR_CARRY_RESULTS.md, update current handoff and stop. No automatic repair/retry/sweep or follow-up training after outcome.
