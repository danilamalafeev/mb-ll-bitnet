# E15 — V7 implementation review

Статус: **CLEAR для запуска зарегистрированного pilot**. Neural training и
scientific inference не запускались.

Проверен designated frozen preflight:
`runs/register_e15_preflight/v7/manifest.json`. Canonical manifest hash:

`bdbb471f1ada73e1ef50b9b6b6cc768bdcd94dc4bad90de12e8eb5a5e24e9c74`.

`_verify_frozen_sources(manifest)` прошёл; все hashes V7 совпадают с текущими
module, runner, tests, protocol, audit, model, quantization и runtime files.
V5 и V6 сохранены как прежние provenance artifacts и в pilot не используются.

Независимые проверки:

- `pytest -q tests/test_register_e15.py`: **9 passed**;
- `python -m py_compile` для module, runner и runtime: passed;
- direct preflight path: passed;
- V7 manifest schema и source verification: passed;
- фактические parameter counts: QAT `152768`, GRU `152720`.

Научные execution paths соответствуют зарегистрированному repair: final CE
tie-break усредняется по длинам; paired QAT/GRU wins, losses и ties считаются
по каждому состоянию; gate принимает ровно полный набор 32 seen-программ,
проверяет `states==32`, bounded integer joint counts и primitives `32/32`;
оба seed-0 arms записываются до gate, failure останавливает дальнейшую
inference. Selected checkpoints проверяются по manifest/source hash, seed,
update, tag, config, prefix digest и model digest. Primary, secondary,
symbolic и causal controls, а также primary predicates записываются в report.

Ключевая V7-проверка recovery закрыта: после успешного обучения шести arms
runner атомарно сохраняет `training_complete` до начала evaluation. Forced
evaluation failure оставляет все шесть arm records и selected checkpoints,
а `--eval-only` восстанавливает evaluation read-only без повторного обучения
и без обхода seed-0 gate. Это проверено focused test.

Clearance относится только к запуску с V7 manifest и текущим source snapshot.
Этот review не сообщает gate outcome, primary accuracy, comparison result или
какое-либо научное преимущество QAT.

## Current source hashes

`looped_bitnet/register_e15.py` —
`bf3e7fa45dfaf2226cd8b1e4541afce36d797ac69f8ad7c0a975eba041561dac`.

`scripts/register_interpreter_e15.py` —
`66898ad816e4f74962ab7f64da23b33b55be6c41a41bd0a67c00d3dd58acbef3`.

`tests/test_register_e15.py` —
`08cc084a281d18e18f8143850bdca03046fec2e8089984c3c2b2723745d5dd6f`.

`looped_bitnet/runtime.py` —
`33382f4ee28592d4fdba01d26913c46e313054a481c5a1f4f4b069cdbc58377d`.

`runs/register_e15_preflight/v7/manifest.json` — file SHA256
`72e240e2f70d32592ff8bc2d5ffeb851580cc62504af66c4850b3a2ea590f733`.
