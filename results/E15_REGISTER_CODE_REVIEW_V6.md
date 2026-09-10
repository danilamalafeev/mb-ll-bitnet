# E15 — V6 implementation review

Статус: **CLEAR для запуска зарегистрированного pilot**. Neural training и
scientific inference по-прежнему не запускались.

Проверен designated frozen preflight:
`runs/register_e15_preflight/v6/manifest.json`. Его canonical manifest hash:

`bd84ab6eec4b682adb56421da4b6beebf1c420a524991545d1e6725e08474d09`.

Проверка `_verify_frozen_sources(manifest)` прошла. Все source hashes в V6
совпадают с текущими файлами, включая runner и tests; прежний V5 намеренно
сохранён как устаревший provenance artifact и в pilot не используется.

Финальные проверки:

- `pytest -q tests/test_register_e15.py`: **8 passed**;
- `python -m py_compile` для E15 module, runner и runtime: passed;
- documented direct preflight path: passed;
- manifest schema, source set и schedule: passed;
- QAT parameter count `152768`, GRU parameter count `152720`.

Проверенные scientific paths соответствуют repair note: final-position dual
CE selection macro-averaged by program length; true per-state paired
QAT/GRU wins, losses and ties; complete 32-row gate; seed-0 gate stop before
any test inference; selected checkpoint, prefix digest, source hash, seed,
config and model digest guards; read-only six-arm recovery; primary,
secondary and symbolic/causal controls in the completed report.

Clearance applies only to the registered execution path using V6 and its
source snapshot. No result, gate outcome, superiority claim or scientific
inference is implied by this code review.

## Current source hashes

`looped_bitnet/register_e15.py` —
`26454a7fe08c1203d3343528b4d1262102c0c05d3634459d531a650566a0ed50`.

`scripts/register_interpreter_e15.py` —
`63be107c372898edf40b6d2871281d66bfde91483e9f9187f6fcf645c6c07403`.

`tests/test_register_e15.py` —
`730ea54a6aa61a80dba12767d1d5ec4e7ad20f8f65951a67cacdd16fdb64c9a1`.

`looped_bitnet/runtime.py` —
`33382f4ee28592d4fdba01d26913c46e313054a481c5a1f4f4b069cdbc58377d`.

`runs/register_e15_preflight/v6/manifest.json` — file SHA256
`5344a924b85b566511879ef8271bdbf58336ce7e5caf919704c69224351be915`.
