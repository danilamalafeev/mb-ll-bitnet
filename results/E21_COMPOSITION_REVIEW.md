# E21 independent result review — ACCEPT

2026-09-07; independent Astra/low. Root scientific report `runs/e21_composition/report.json` SHA256 `121fa9094bfb616d8d267bf27b624de25fd63dd94800aaf7b6b3c57e32287cc7`. Reviewer performed one strict reload of the frozen E20 u8000 model and exactly seven256-state program forwards, then stopped. All1792 per-state prediction/target traces and report fields reproduced exactly. Verification elapsed14.997s including loading, checks and independent recounting; not a training/runtime comparison.

| Primary program | Final joint /256 | >=244 |
|---|---:|---|
| ADD → XOR |256|pass|
| ADD → ADD → XOR |253|pass|
| ADD → XOR → ADD |254|pass|
| ADD → XOR → SWAP |255|pass|
| XOR → ADD → XOR |253|pass|
| SWAP → ADD → XOR |251|pass|

Six-row conjunction passes. Descriptive primary total1522/1536 (99.0885%); reserved initial-state stratum187/192. Every first/second prefix is correct; observed failures are at the third instruction. Separate equivalent ADD → XOR → XOR control252/256; it is excluded from the primary predicate and aggregate.

Independent integer ADD-modulo16/XOR/SWAP enumeration reproduced every target trace and final truth table: six distinct primary maps absent from all32 seen programs; control mapping equals ADD. Direct recounting from predicted/target pairs reproduced final joint/x/y, every prefix and full trace for all256 and each frozen192/32/32 stratum. Frozen manifest equality, exact reviewed evaluator/test hashes, model/checkpoint identity, source hashes and all195 protected files verified before/after replay. No report/source/checkpoint changes.

Checkpoint SHA256 `4507dd19d95b8a90f6ee1b487abc6493a26ba19700b784eaa2949dbfaa75c85d`; reviewed runner `ad3c3e89d79b57389b3d9a2f48a51543df409354a0d50dcd249f52aab7dc8948`; tests `39106309e8483e57ede268c86308c5a78d8fceb975759a14d00172c00f7d1fff`.

Cost separation: root scientific evaluation7 forwards/1792 cases/5120 readouts/40960 native substeps,0 training. This independent replay adds exactly the same verification cost (7/1792/5120/40960),0 training. Pre-clearance code QA is separate: reviewer test replay had7 zero-logit toy forwards plus one real untrained ADD forward on2states (16 real native state-substeps); supplemental synthetic gate checks used14 toy ADD-only forwards. Toy nominal substep arithmetic does not denote executed trained recurrence. Executor QA is separately owned in root accounting.

The registered numerical single-model predicate is met by frozen seed0 float core/signed-bit input/native8/u8000. The model and architecture were adaptively selected using earlier validation evidence. This is not the original E15 paired QAT/GRU three-seed claim, an independent seed replication, or evidence of arbitrary-length reasoning. No new training, programs, checkpoints, seeds, tuning or next experiment is authorized by this acceptance. Registered E21 stop is satisfied after root documentation.
