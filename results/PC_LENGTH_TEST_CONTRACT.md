# PC: inference-only length diagnostic contract (proposal)

2026-09-09. Owner: transfer_design (Astra/low); root owns shared documentation. Status: DESIGN ONLY, pending target environment and independent implementation review. No data generation, checkpoint loading, model forwards or training were performed to write this document. This diagnostic concerns the existing finite DSL; it does not validate the wider language-model vision.

## Fixed endpoints and migration gate

Use exactly 12 existing h128 endpoints: seeds 0/1/2 × float/W4 × A8000/B8000 (absolute u40000). Seed0 comes from accepted E36; seeds1/2 from accepted E38. B8000 is the primary equal-added-update comparison. B_match is excluded: equal-training-step benefit already has accepted evidence, and adding it would increase this diagnostic by 50%. No weights, optimizer, training streams, architecture, native8 or readout behavior change. Zero training, including QA.

Transfer inventory must include the strict loader's transitive source, checkpoint, manifest, runtime-lineage and reference dependencies, not just 12 endpoint files. Preserve accepted bytes and hashes. Use a separately reviewed path-root adapter if required; do not bypass identity checks or rewrite original provenance. Validate each dependency once per process and reuse an immutable validated context; load each endpoint once and execute its complete pool. Do not repeatedly rebuild old training manifests to serve individual programs. Any new caching/path adapter requires pure tests and narrow review.

Before new science, target CPU must decode one fixed existing accepted L7 program on all256 initial states for each endpoint and match every saved trace/target exactly. Select the lexicographically first existing allowed primary L7 program by its stored ID before reading predictions. This is a small migration gate, not a rerun of historical evaluation. Preserve original CPU dtype/determinism settings where available; record runtime versions, actual threads and time. A mismatch blocks new science for investigation; do not silently accept tolerances on discrete outputs.

CUDA is optional and unverified. First establish GPU, driver and PyTorch availability. If desired, run the same 12 fixed migration batches on CUDA, comparing CPU decoded traces exactly and recording maximum absolute/relative logit differences without claiming bitwise equivalence. A decoded mismatch blocks CUDA for this protocol; retain CPU as the valid route. Passing this finite gate is limited numerical evidence, not proof of equality at L32. Never select a backend based on scientific accuracy. Backend and any dtype/kernel change must be frozen before science; no automatic precision conversion or GPU tuning sweep.

## 1. Exact-function padding at even lengths

The old E35/E37 L3 bases plus SWAP pairs reach only odd lengths. Use explicit even bases instead: take each P in {(ADD,ADD),(XOR,SWAP),(SWAP,XOR)}, append one fixed SWAP to P to form Q of length3, and choose O in {ADD,XOR,SWAP}. Evaluate Q→(SWAP→SWAP)^k→O with k={0,4,6,10,14}, giving lengths {4,12,16,24,32}: nine fixed families,45 programs per endpoint. L4 is the paired anchor; do not compare these traces directly against old L3 anchors.

The extra fixed SWAP is present in every family member, so padding preserves each family's full-domain function and pre-O true state. Prove both on all256 states before model work. Families need not be semantically distinct; report actual full-map multiplicity. Classify each anchor against training-prefix functions rather than assuming it is seen. Additional SWAPs change both history and compute; this is not a pure hidden-state causal intervention.

## 2. New compositions at L12/16/24/32

Propose six programs per length,24 total, each run on all256 states. Before any forward, construct a deterministic candidate stream with seed20260909 and a documented fixed PRNG/opcode order. Inspect at most1024 distinct candidate strings per length (4096 total), excluding ADD→XOR to retain the previous allowed-composition regime. Candidate generation itself must have a finite attempt bound, proposed4096 draws per length; duplicates count toward that draw cap.

Compute each candidate's exact map over all256 states. Select the first six qualifying maps per length that are absent from the accepted union of all trained prefix functions (94, verify against manifests), absent from the previously opened E36/E38 primary function pool (23, verify), and distinct within that length. Preserve the complete selected strings/maps and generation accounting. Report cross-length map overlap explicitly; do not call new strings independent functions. Semantic novelty is relative to those finite sets, not a proof that no shorter equivalent exists.

If six qualifying maps cannot be found at any length within the cap, stop before scientific freeze and report the shortfall. Do not claim that no such functions exist, launch an exhaustive search, change the DSL, relax novelty or refill using observed model errors. A smaller pool needs an explicit superseding contract and exact budget. Full-domain maps establish class identity; neither sampled candidate search nor existing results establish minimal program length32.

The padding suite supplies equivalent-function long execution; the new-composition suite supplies functions not previously exposed in training or the accepted primary evaluation pool. Keep these categories separate in reports. Initial-state192/64 strata inherit accepted lists and concern initial states only, not unseen intermediate states.

## Budget and outputs (proposed maxima)

A forward here means one complete program evaluated in one batch; native steps mean examples × program length ×8. If memory requires chunking, review a new forward-call budget before launch; case/native totals remain fixed. CPU is baseline; backend choice does not double scientific execution.

| Stage | Forwards | Cases | Native steps |
|---|---:|---:|---:|
| Tiny inference QA: one endpoint,2 states at L4 and L32 |2|4|576|
| Required CPU migration:12 endpoints × one L7 ×256 |12|3072|172032|
| Optional CUDA migration: same frozen batches |12|3072|172032|
| Padding science:12 ×45 ×256 |540|138240|19464192|
| New-composition science:12 ×24 ×256 |288|73728|12386304|
| Independent narrow replay:12 × one frozen L32 ×256 |12|3072|786432|
| Maximum including optional CUDA |866|221188|32981568|

Science alone:828 forwards,211968 cases,3981312 instruction readout positions,31850496 native steps. All stages:0updates. One forward returns both register predictions at every instruction; readout positions count instructions, not register scalars. Tiny QA uses the first padding family, fixed first2 state IDs; reviewer replay uses its L32 member, selected prospectively. Raw independent audit covers ALL saved cases; the narrow replay is not mislabeled a full exact replay. Record attempted/completed counters, deserializations, wall time and failures separately; a failed attempt is not free. A read-only hash inventory does not count as a checkpoint deserialize.

Outputs: fresh versioned directory selected by root, immutable protocol/source/data/checkpoint manifest, separate runtime/accounting, complete decoded traces and DSL targets, one results report and one independent review. Preserve accepted artifacts. CPU migration consumes accepted saved predictions; do not regenerate historical reference outputs. New data and code require pure hand-oracle tests → tiny QA → independent CODE CLEAR → one freeze/preflight → one detached science supervisor and one completion wake. No polling, automatic retry, training, extended length, new seed or follow-on sweep.

## Analysis and stop

Report final and full-trace errors/counts separately for every length, precision, seed, branch, suite and inherited initial-state stratum. Pair A versus B on identical program/state IDs: recovered, introduced, both-correct, both-wrong; preserve program-level counts. Within padding, pair each long member with its own L4 anchor and report degradation/recovery. Final-O accuracy conditional on BOTH paired pre-O readouts correct is supplemental and must report excluded counts.

Descriptive proposed questions: how far does B retain its advantage, and does degradation differ between equivalent functions and new semantic compositions? No new threshold, seed averaging, p-value or arbitrary-length success claim. Explicitly report zero-error A ties; neither an L32 pass nor failure proves a universal algorithm or architectural impossibility. Do not mix equal-update A/B with equal-inference-compute claims: longer programs cost more inference steps.

CURRENT STOP: setup pending target execution/transfer environment. This document does not authorize implementation expansion or execution by itself. Once environment is established, root can proceed through the existing authorized workflow, freezing this proposal or a reviewed superseding contract before any scientific forward.
