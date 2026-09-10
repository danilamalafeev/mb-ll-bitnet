# Pair-carry intervention: all36 failures repaired, zero regressions

2026-09-10. Verdict: **ACCEPT; prospective positive-pilot criterion met.**

At the accepted latent local2000 endpoint, replacing only the carried latent state after each registered identity SWAP pair with that pair's entry state corrected every previously failed focus trace. Both ordinary SWAP operations and their logits were still computed; the final unpaired SWAP and useful suffix remained ordinary. The sham control exactly reproduced all saved accepted focus traces. No training occurred.

| Focus | Sham full/final correct | Pair-carry full/final correct | Repairs | Regressions |
|---|---:|---:|---:|---:|
| L24 | 2286/2304 | 2304/2304 | 18 | 0 |
| L32 | 2286/2304 | 2304/2304 | 18 | 0 |
| Total | 4572/4608 (99.21875%) | 4608/4608 (100%) | 36 | 0 |

All4572 originally correct traces stayed correct. The36 original failures were repaired in both full trace and final answer. Wrong instruction positions fell from228 padding plus36 useful-suffix positions to0. Prefix errors remained0. Train-stratum repairs:30/3456cases; validation:6/576; test:0/576, because test had no original failures. Validation+test combined:1146/1152 to1152/1152. These strata belong to the already-opened pool, not a newly untouched evaluation.

## What this establishes

Under identical weights, inputs and ordinary computations, resetting the state carried across semantically identity SWAP pairs was sufficient to eliminate the observed long-padding failures in this fixed pool. This supports a causal contribution from accumulated carried-state changes to these failures. It complements the earlier observation that simple monotonic norm growth, a large jump or a dominant channel did not explain them: harmful changes need not produce those magnitude signatures.

The intervention uses known program structure and an external copy of pair-entry z. It is a diagnostic control, not an autonomous learned architecture or a proposed production inference rule. It does not uniquely locate the defect in writer W: reader/core/writer/readout interactions remain possible. It neither proves a stable vector cycle nor establishes arbitrary-length or unseen-program generalization. The36 failures repeat two semantic prefix-end states(4,14)/(14,4) across prefixes, lengths and suffixes; they are not36 independent discoveries. No extra seeds, new programs, new training or composition-suite evaluation were performed here.

A sensible next research question is whether training can encourage preservation of state through identity compositions without an external pair-aware copy, while retaining nonidentity composition performance. This is a proposed direction only; no additional experiment is launched or budgeted in this result.

## Verification and cost

Independent stdlib audit reconstructed DSL targets, hash-ranked strata, exact18program ×256state coverage in both arms, sham equality to saved baseline, full/final paired counts, first-error positions and role-specific errors directly from36saved row files. It used0model calls and0checkpoint deserializations. Root verified96source/evidence bindings, all source-gate hashes, report/accounting consistency and empty stderr/supervisor logs. The background completed exit0 and queued one notification. Runtime aggregate tables pool both arms; this report uses the independent per-arm audit instead.

Science actual attempted/completed:36forwards,9216cases,258048instruction positions,2064384forward native steps, one strict endpoint load and3underlying deserializations;0backwards, optimizer steps or updates, no failures. Background wall time68seconds includes setup/validation/output; it is not isolated model latency. Separate accepted QA cost:3forwards,6cases,30positions,240native steps,3deserializations,0updates. Combined QA+science:39forwards,9222cases,258078positions,2064624steps,6deserializations,0updates. No QA repetition, scientific retry or model replay was used for acceptance.

QA verified exact preservation of model/adapter/optimizer/RNG/mode. Science completed through the same accepted state-preservation context; it did not save an additional standalone before/after state-digest report. The endpoint and accepted source files remain unchanged.

## Artifacts

Code/artifact root: `C:/Users/я/LoopedBitNet_AI2_inference/project`.

- Protocol: `../results/PC_LATENT_SLOTS_PAIR_CARRY_PROTOCOL.md` in this documentation project.
- Raw rows and immutable input/source bindings: code-root `runs/pc_latent_slots_pair_carry_v1/science/`.
- Independent raw audit: code-root `runs/pc_latent_slots_pair_carry_v1/science_saved_audit.json`.
- Acceptance with verified hashes of all36output rows: code-root `runs/pc_latent_slots_pair_carry_v1/science_accept.json`.
- Independent audit source: code-root `scripts/audit_pc_latent_slots_pair_carry.py`.
- Accepted QA and source gates remain in code-root `runs/pc_latent_slots_pair_carry_v1/`.

Registered scope complete. STOP; no further runs, retries or training.
