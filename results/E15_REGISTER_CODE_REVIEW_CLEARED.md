# E15 — final implementation review

Статус: **BLOCKED — source freeze mismatch**. Neural training и scientific
inference не запускались.

Исправленный runner закрывает проверенные ранее дефекты: validation selection
использует final-position dual CE, усреднённый по длинам; paired QAT/GRU
outcomes считаются по каждому из 256 состояний; gate требует полный набор 32
seen-программ; seed-0 arms завершаются до gate, а failure останавливает primary
и secondary inference; selected checkpoints проверяют manifest, source hashes,
seed, update, tag, config, prefix digest и model digest; recovery read-only и
не обучает; controls, primary predicates и secondary evaluation записываются.

Независимые проверки текущего рабочего дерева:

- `pytest -q tests/test_register_e15.py`: **8 passed**;
- `python -m py_compile` для E15 module, runner и runtime: passed;
- documented direct preflight command: passed;
- свежий preflight source verification: passed;
- фактические parameter counts: QAT `152768`, GRU `152720`;
- schedule: `666/668/666`, 2000 updates, suffix `(2, 2)`.

Блокирующее расхождение относится к frozen provenance. Указанный frozen
`runs/register_e15_preflight/v5/manifest.json` имеет canonical manifest hash
`5d54baecadf422e6f8047c6c32bf7991004ba530178912b182730fe49495352e`, но
содержит hashes прежних файлов: runner
`1b556a1eebe484cfeb8b4543ec504d7c2ab5b6000f24454bfdff069a914740e8`, tests
`513c19513e8bd9f078ec3749167c21676f47d6b25f5c5feb617e8852a6d2d3fd`.
Текущие проверенные файлы имеют runner
`63be107c372898edf40b6d2871281d66bfde91483e9f9187f6fcf645c6c07403` и tests
`730ea54a6aa61a80dba12767d1d5ec4e7ad20f8f65951a67cacdd16fdb64c9a1`.
Поэтому `_verify_frozen_sources(v5)` корректно завершится ошибкой до первого
update. Свежий preflight с текущим деревом даёт canonical hash
`bd84ab6eec4b682adb56421da4b6beebf1c420a524991545d1e6725e08474d09` и
подтверждает все source hashes.

Нужно заново материализовать designated preflight после окончательной
фиксации файлов и записать его новый canonical hash, либо восстановить ровно
тот source snapshot, на котором был создан v5. До этого implementation review
не может выдать CLEAR.

## Текущие source hashes

`looped_bitnet/register_e15.py` —
`26454a7fe08c1203d3343528b4d1262102c0c05d3634459d531a650566a0ed50`.

`scripts/register_interpreter_e15.py` —
`63be107c372898edf40b6d2871281d66bfde91483e9f9187f6fcf645c6c07403`.

`tests/test_register_e15.py` —
`730ea54a6aa61a80dba12767d1d5ec4e7ad20f8f65951a67cacdd16fdb64c9a1`.

`looped_bitnet/runtime.py` —
`33382f4ee28592d4fdba01d26913c46e313054a481c5a1f4f4b069cdbc58377d`.
