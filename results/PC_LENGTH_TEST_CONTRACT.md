# PC: inference-only length diagnostic contract

2026-09-09. Status: CUDA science complete after target-environment validation, frozen preflight, and independent raw-output review. This diagnostic uses the existing finite DSL. It performs zero training and does not validate the wider language-model vision.

## Fixed endpoints and migration gate

Use exactly 12 existing h128 endpoints: seeds 0/1/2 x float/W4 x A8000/B8000 (absolute u40000). Seed0 comes from accepted E36; seeds1/2 come from accepted E38. B8000 is the primary equal-added-update comparison. B_match is excluded. No weights, optimizer, training streams, architecture, native8 or readout behavior change.

Preserve accepted checkpoint, manifest, runtime-lineage and source bytes and hashes. Use a separately reviewed Windows path/resource adapter when required; do not bypass identity checks or rewrite original provenance. Validate each dependency once per process, load each endpoint once, and record deserializations and failures separately.

Before new science, target CPU must decode one fixed existing accepted L7 program on all 256 initial states for each endpoint and match every saved trace/target exactly. Select the lexicographically first existing allowed primary L7 program by stored ID before reading predictions. This is a migration gate, not a historical evaluation rerun. Preserve original CPU dtype and determinism settings where available. A mismatch or missing saved trace/target blocks new science.

CUDA is optional. First establish GPU, driver and PyTorch availability. If used, run the same 12 frozen migration batches on CUDA, compare decoded traces exactly with CPU, and record maximum absolute/relative logit differences without claiming bitwise equivalence. A decoded mismatch blocks CUDA and CPU remains the valid route. Freeze backend and dtype before science; do not tune precision or kernels and never select a backend from science accuracy.

## Exact-function padding at even lengths

The old E35/E37 L3 bases plus SWAP pairs reach only odd lengths. Use explicit even bases: for P in {(ADD,ADD),(XOR,SWAP),(SWAP,XOR)}, append one fixed SWAP to form Q of length 3, then choose O in {ADD,XOR,SWAP}. Evaluate Q -> (SWAP -> SWAP)^k -> O with k={0,4,6,10,14}, giving lengths {4,12,16,24,32}: nine fixed families and 45 programs per endpoint.

The extra fixed SWAP is present in every family member. Prove full-domain function and true pre-O state identity on all 256 states before any model forward. Report actual full-map multiplicity and classify anchors against training-prefix functions. Additional SWAPs change both history and compute; this is not a pure hidden-state intervention.

## New compositions

For lengths 12/16/24/32, select six programs per length, 24 total, using a deterministic candidate stream with seed 20260909 and documented PRNG/opcode order. Exclude ADD -> XOR. Inspect at most 1024 distinct candidate strings per length and at most 4096 draws per length; duplicates count toward the draw cap.

Compute each candidate's exact map over all 256 states. Select the first six qualifying maps per length absent from the accepted union of all trained prefix functions (94), absent from the previously opened E36/E38 primary function pool (23), and distinct within that length. Preserve selected strings/maps and generation accounting. Report cross-length map overlap. If a length cannot produce six qualifying maps within the cap, stop before scientific freeze; do not relax novelty, refill from model errors, or run exhaustive search.

## Budget and outputs

| Stage | Forwards | Cases | Native steps |
|---|---:|---:|---:|
| Tiny inference QA | 2 | 4 | 576 |
| CPU migration | 12 | 3072 | 172032 |
| Optional CUDA migration | 12 | 3072 | 172032 |
| Padding science | 540 | 138240 | 19464192 |
| New-composition science | 288 | 73728 | 12386304 |
| Independent narrow replay | 12 | 3072 | 786432 |
| Maximum including optional CUDA | 866 | 221188 | 32981568 |

Science alone is 828 forwards, 211968 cases and 31850496 native steps. All stages are zero updates. One forward returns both register predictions at every instruction. Record attempted/completed counters, deserializations, wall time and failures separately.

Outputs belong in a fresh versioned PC-only directory: immutable protocol/source/data/checkpoint manifest, separate runtime/accounting, complete decoded traces and DSL targets, one results report and one independent review. Preserve accepted artifacts. Use pure hand-oracle tests -> tiny QA -> narrow independent review -> one freeze/preflight -> one CUDA science supervisor and one completion wake. No polling, automatic retry, training, extended length, new seed or follow-on sweep.

## Analysis and stop

Report final and full-trace errors separately for every length, precision, seed, branch, suite and inherited initial-state stratum. Pair A versus B on identical program/state IDs and preserve program-level counts. Within padding, pair each member with its L4 anchor. Conditional final-O accuracy is supplemental and includes excluded counts.

Do not claim universal algorithms, architectural impossibility, arbitrary-chain execution, bitwise CPU/GPU equality or success beyond the finite protocol. Do not mix equal-update A/B with equal-inference-compute claims. Stop after fixed artifacts and review, regardless of effect size.

## Current stop condition

The CUDA migration and fixed science suites are complete. Results are frozen under `runs/pc_inference_v1/science_cuda_v1/`; no training, retry, extended-length sweep, or new-seed run was performed.
