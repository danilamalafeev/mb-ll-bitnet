# Gated carry pilot v1 — prospective paired protocol

2026-09-10. Root owner01a08b1b-e103-77b3-b360-5734cf71e230. User approved testing residual/gated state updates after rejecting a SWAP-specific loss. This is a new bounded training experiment, not a continuation of the completed pair-carry scope. No new model run has begun.

## Hypothesis and minimal architectural change

The existing writer replaces the entire latent state every instruction. Test whether a direct learned carry path reduces destructive accumulation while preserving useful operations. Both arms start from the accepted float seed0 latent local2000/absolute42000 endpoint, code-root runs/pc_latent_slots_v1/science/checkpoints/local2000.pt, SHA256 bb7d4ea2739d97ccd9b43663657460c5f775bc8514543b71e8a14e88e03417d5. A uses the current writer unchanged. B adds a two-output scalar-per-slot gate:

proposal = existing W(normalized returned h), reshaped [batch,2,16]
g = sigmoid(Linear(concat(flatten(z), normalized returned h))), shape [batch,2,1]
z_next = (1-g)*z + g*proposal = z + g*(proposal-z)

This is the minimal gated residual proposal, not a separate new delta network. Two gates share all16channels within each slot; Linear(160,2,bias=True) adds322trainable parameters. Initialize weights exactly0 and both biases log(9), giving g=0.9 initially. Preserve global RNG during construction. This fixed initial interpolation may itself change accuracy; initial evaluations separate that effect from subsequent learning. No tuning of initial gate or coefficients after seeing results.

Gate receives only current z and ordinary normalized hidden features, never opcode IDs, position, pair markers, identity labels, targets or original registers. The ordinary core still sees its usual opcode. Both slots remain unconstrained continuous vectors. No oracle carry, pair restoration, slot alignment, SWAP/XOR loss or auxiliary target is added. Both arms use the same ordinary per-position x/y CE, native8 core, reset h/KV, original reader/writer and existing training numerical utilities. Gate is applied after ordinary current logits, affects future state, and must preserve cross-step autograd without detach. Finite-check raw proposal, gate and actual carried state.

Use scoped adapter-reader prehook to capture current z and writer forward hook to blend after computing ordinary proposal; call the accepted latent forward unchanged. Hooks removed on success/failure. Gate may be registered as adapter.carry_gate so existing adapter state/gradient traversal includes it, but existing accepted adapter identity/load gates must run before attaching the new module. Define new versioned architecture identity for B and include all gate names/state in snapshot identity. Never monkeypatch accepted modules globally or edit accepted source.

## Matched training

Fresh A and B load the same strict endpoint including optimizer/RNG. Original optimizer groups/moments/hyperparameters unchanged. B appends one named gate-only group copying adapter group's hyperparameters; its322parameters have no historical moments. Record extra capacity and this asymmetry. Train all model/adapter/gate parameters used by the forward, clip combined norm at1 with error_if_nonfinite=True. No gate-only training or altered learning rate.

Each arm gets exactly1000updates, batch64, first1000batches of saved accepted science_stream.json: lengths(1,2,3,4,5,6)*166+(1,2,3,4), sum3496. Reuse exact programs/states/targets/order; no new sampling or training curriculum. Both start at the same saved RNG and use their own optimizer. Record immutable stream hashes and prefix identity. This is another continuation on an opened training stream, not a from-scratch claim. Fixed endpoint local+1000/absolute43000, no early stopping/tuning.

## Evaluation scope

Old69program suite/all256states: reuse the hash-bound accepted initial A predictions from the saved local2000 final evaluation, without replay. Evaluate B initially and both arms finally,3x69forwards. Report old padding and useful composition separately; the original composition score is6144/6144.

New81program control suite: three prefixes ADD ADD; XOR SWAP; SWAP XOR. Three identity cells SWAP SWAP SWAP SWAP; XOR XOR XOR XOR; SWAP XOR XOR SWAP. Repeat each cell2,4,8times, then one suffix ADD/XOR/SWAP. Cartesian3prefix x3cells x3repeats x3suffix=81programs,27at each length11/19/35, all256states. Every cell is semantically identity; ordinary intermediate targets must still be correct. No oracle/pair-aware behavior at inference. Validate all cell identities over256states with pure DSL oracle before model work. Freeze IDs/programs/state strata; do not inspect model results before freeze. These are new syntactic programs at longer lengths, but deliberately related finite semantic controls, not independent arbitrary-program evidence.

Evaluate new81 at initial and final A and B,4x81forwards. No extra intermediate evaluations. Same hash-ranked train192/validation32/test32 states and heldout64 aggregation. Persist decoded/DSL traces with exact IDs/state joins, full/final scores, per-suite/cell/length/stratum, repairs/regressions, first error/recovery. Gate mean/min/max per slot and opcode/position can be computed during these existing passes only; descriptive, not semantic slot labels. No full hidden tensor dump or probes.

Primary positive pilot: final B has strictly fewer full-trace errors than final A on new81 overall AND on old padding, with zero B-wrong/A-correct cases on old useful compositions AND B retaining all6144/6144original useful-composition full traces. Report per-cell regressions even when the total improves. Compare each final to its own initial: if B's advantage already exists initially, attribute that portion to the fixed carry initialization, not learned gating. If both arms reach a ceiling, report tie rather than success for B. Because gate and original weights train jointly, this pilot compares architectural training packages; it cannot uniquely attribute a final advantage to learned gate values rather than fixed carry plus adaptation of the other weights. No post-hoc threshold selection, seed extension or coefficient search. Any mismatch/nonfinite/failure stops without rescue tuning.

## Stages, QA and checkpoint integrity

First deliverable only new scripts/pc_gated_carry.py plus focused pure tests: exact322count/init/RNG preservation; hand-calculated convex blend and slot broadcast; unchanged current readouts/future-only effect; no target/opcode/pair gate input; no detach and finite nonzero final-only-loss gradients to earlier z and gate; raw nonfinite rejection before blending; hooks cleanup; gate-only optimizer group association; exact identity-cell semantics and coverage/budget; repair/regression/tie oracle. No real model/checkpoint/CUDA work in this stage.

After pure acceptance, implement disposable QA only. For each arm: first2examples of batch0/L1 update once; save complete snapshot; batch1/L2 update once; reload snapshot and repeat the identical L2 update once. Exactly3updates perarm,6total forwards/backwards/updates,12cases,20positions,160native steps. Two strict endpoint loads=6underlyingdeserializations, plus2snapshot reloads=8total expected; track actual counts. Compare exact model/adapter/gate/optimizer/moments/CPU+CUDA RNG/modes/names after uninterrupted/reloaded next update; check finite loss and gate gradients. QA snapshots must include gate parameters/group names and resume index. No extra evaluation or numerical probes; QA arms discarded.

After QA ACCEPT, prepare new science-only runtime with a new immutable freeze (source/protocol/accepted QA/stream prefix/eval scope/parent/baseline hashes) before model construction. Reuse accepted strict loader, settings and training utilities; no original full preflight reconstruction or QA rerun. Science one endpoint load perarm=6deserializations expected, no checkpoint reload. Model/adapter/gate/optimizer/RNG/mode preserved across evaluation. Save complete atomic checkpoints at0,250,500,750,1000 perarm, including all state and exact next stream index plus provenance. Preserve previous checkpoints; no resume/retry without separately registering recovery costs.

Track actual attempted/completed forward/backward/update/deserialization phases separately, and preserve failures/partial costs. Native steps count forward core work only, not backward FLOPs/gate overhead. New code only, accepted checkpoints/source/results unchanged. Fresh output runs/pc_gated_carry_v1/{qa,science}; root writes shared docs. One detached run_and_wake supervisor per QA/science stage to owner task, no model polling. Separate saved-artifact verification uses no model replay. Final report and STOP after fixed pilot; no automatic deployment/push/new experiment.

## Fixed science budget

Training:2000forwards/updates,128000cases,447488positions.
Evaluation:531forwards,135936cases,2792448positions (old suite3x69, new suite4x81).
Total:2531forwards,263936cases,3239936positions,25919488forward native steps;2000backwards/optimizer updates;2strict endpoint loads/6deserializations expected.
QA separately:6forwards12cases20positions160steps6updates8deserializations expected. All numerical choices prospective and fixed; report runtime and actual counters separately.
