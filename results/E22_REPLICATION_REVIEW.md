# E22 independent result review — ACCEPT; replication predicate failed

Reviewer Astra/low,2026-09-07. Both registered seed1/seed2 runs completed
correctly and their evidence is accepted. The strict primary replication
predicate is FALSE for both seeds and therefore FALSE jointly. This is a valid
negative replication outcome, not a technical blocker or missing experiment.
No additional training, retry or evaluation set was used by this review.

## Verified primary outcomes

Threshold is EACH primary program at least244/256; an average cannot rescue
one failure. Control is excluded from the primary conjunction.

| Primary program | Historical seed0 | Seed1 | Seed1 pass | Seed2 | Seed2 pass |
|---|---:|---:|---|---:|---|
| ADD → XOR |256|255|yes|253|yes|
| ADD → ADD → XOR |253|233|no|248|yes|
| ADD → XOR → ADD |254|243|no|241|no|
| ADD → XOR → SWAP |255|248|yes|250|yes|
| XOR → ADD → XOR |253|244|yes|248|yes|
| SWAP → ADD → XOR |251|240|no|246|yes|

Separate ADD→XOR→XOR control: seed1=246/256, seed2=248/256. Its scores do not
enter the six booleans or their conjunction. All primary/control predictions
and train192/validation32/test32 strata were retained for both seeds.

| Predicate | Seed1 | Seed2 | Both-seed conjunction |
|---|---|---|---|
| Seen validation prerequisite |false|true|false|
| Six-primary predicate |false|false|false|
| Prerequisite AND primary |false|false|false|

Seed1's prerequisite failure is not just ADD31/32. Six seen compositions also
fall below31/32: ADD→SWAP30, ADD→ADD→ADD29, ADD→ADD→SWAP30,
ADD→SWAP→ADD29, ADD→SWAP→XOR30 and SWAP→ADD→ADD30. Other imperfect seen
compositions at31/32 do meet their numerical threshold. Seed2 has all primitives
32/32 and every seen composition at least31/32. Its failed primary
ADD→XOR→ADD241/256 therefore remains a primary failure despite meeting the
seen prerequisite. Seed1's primary cases were evaluated despite its prerequisite
failure exactly as prospectively authorized; no row was selectively omitted.

## Independent checks

Strict-loaded both u8000 finals against the frozen E22 preflight, with exact
seed/native8/initial/config/source/reference/stream provenance, model and
AdamW/RNG digests, tensor inventory and parameter count151232. Both optimizers
contain all32 parameter states and every step counter is8000. Reported initial
RNG digests equal the frozen post-constructor states for their respective seeds.
Preflight reconstruction checks root's independent complete-initialization,
projection and RNG references; data seed remains0 for both runs.

Replayed BOTH final evaluations:128 seen program/split rows plus14 composition
rows,142 program forwards in total. Every stored prediction/target trace,
prefix/final x/y/joint, full-trace count, macro and gap matched exactly.
Independently executed a direct ADD-mod16/XOR/SWAP interpreter and recounted
all integer metrics from reproduced predictions. Every primary/control row's
all256 and train192/validation32/test32 stratum metrics matched, with exact
stratum membership. Independently recomputed each244 threshold, prerequisite,
primary/control separation, per-seed conjunction and joint report predicates.

Verified all8000 repeated-stream and target digests, seed routing, complete
final/per-seed reports and exact32 finite progress observations per seed. Each
seed recorded8000 updates,512000 examples and8192000 recurrent training
substeps. Both final checkpoint byte hashes remained unchanged across replay:

- seed1: `c6746e633dc27d04bd41c8b7af2039002e4da2bf349e1fe6ae33e492716f06ac`
- seed2: `f8cfe5d89939b0a6722a47460496fd8b49d00178db500aa5a9133e79192b6d29`

Model/optimizer/torch RNG state did not change during evaluation. All206
historical protected file hashes remain unchanged; accepted source and
initialization provenance remain intact. No incomplete seed or scientific
restart is hidden by the complete report.

Read-only independent script `/tmp/e22_result_review.py` passed. Canonical
artifacts: `runs/e22_replication/report.json`, both `seed{1,2}/u8000.pt` and
per-seed reports, plus `runs/e22_replication_preflight/`.

## Work accounting and interpretation

Scientific total:16000 updates,1024000 examples,16384000 training substeps.
Scientific evaluation:17920 program/state cases,46976 boundary readouts and
375808 substeps. This independent replay repeats that inference work as
verification, adding no optimizer updates or new cases.

Stored per-seed training times are91.3907s and97.3751s; scientific inference
1.9406s and2.2385s. Root reports204.52s overall CLI wall time; this review checks
stored work/timing fields rather than independently timing training. Prior
code-clearance QA remains separate: executor63 updates/63 examples/504 training
substeps plus135168 inference substeps; reviewer21 updates/21 examples/168
training substeps plus45056 inference substeps. No result-review training.

Historical seed0 counts above were checked against the accepted E21 report,
not rerun or counted as an E22 replicate. Seeds1/2 vary complete initialization,
including projections, under the same law while holding data and all other
registered settings fixed. The six-program criterion did not survive these
two additional initializations. This weakens a claim of reliable initialization
robustness for the selected setup; it does not show an inability to learn the
operations or identify one causal failure mechanism.

Architecture and evaluation choices were adaptively informed by earlier work;
E21 programs and finite validation data were already open. This is conditional
initialization replication, not independent data-seed replication, untouched
test evidence, the historical paired QAT/GRU result or general reasoning/length
transfer. Both outcomes are complete. The registered stop is reached: no new
seed, tuning, extension, altered stream or extra test follows automatically.
