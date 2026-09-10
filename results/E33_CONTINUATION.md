# E33 — paired seed0 continuation to u32000

Статус: **ACCEPT** после независимого финального ревью; ACCEPT означает корректность артефактов и аудита, а не restoration success.

## Scope

По индексу выбран только seed0 для обоих принятых E32 родителей `float128_seed0/u16000.pt` и `w4128_seed0/u16000.pt`. Каждый получил ровно16000 продолженных обновлений при сохранении модели, AdamW, RNG, stream recipe и режима; финальный абсолютный шаг32000. Архитектура h128/FFN256/head4/native8. Финальная E32-оценка:83 forwards на arm,166 всего.

| Arm | Parent L5 errors | u32000 L5 errors | Strict benefit | L4 errors | L5 restore |
|---|---:|---:|---|---:|---|
| Float128 seed0 | 146/1536 | 166/1536 | false | 5/1536 | fail |
| W4 128 seed0 | 159/1536 | 133/1536 | true | 7/1536 | fail |

Обе модели прошли seen, E21-primary, L4-final и L4-full-trace; обе не прошли L5-final, L5-full-trace и combined gates. Paired training benefit=false. W4−float L5 gap: parent `+13`, final `−33`. Float recovered83 parent errors and introduced103; W4 recovered122 and introduced96.

## Final per-program counts

`final/full` — правильные состояния из256; в последовательностях `A=ADD`, `X=XOR`, `S=SWAP`.

| Arm | L4 programs (final/full) | L5 programs (final/full) |
|---|---|---|
| Float128 | AAAS 256/256; XAAS 256/256; AXAA 255/255; SXAA 255/255; SASX 254/254; XSAS 255/255 | ASAXS 213/211; XAXAS 242/240; XSAAS 227/225; ASASX 217/217; XAAXS 229/229; SASAS 242/242 |
| W4 128 | AAAS 255/255; XAAS 256/256; AXAA 251/251; SXAA 256/256; SASX 255/255; XSAS 256/256 | ASAXS 212/211; XAXAS 242/242; XSAAS 231/231; ASASX 242/242; XAAXS 229/227; SASAS 247/247 |

## Paired own-parent strata

Tuple is `recovered/introduced/both_correct/both_wrong`, keyed by program, state, target and stratum; independent audit also checked full-trace joins.

| Arm | Length | all | train | validation | test |
|---|---:|---|---|---|---|
| Float128 | 4 | 7/5/1524/0 | 3/2/1147/0 | 3/3/186/0 | 1/0/191/0 |
| Float128 | 5 | 83/103/1287/63 | 61/81/971/39 | 9/11/161/11 | 13/11/155/13 |
| W4 128 | 4 | 10/6/1519/1 | 6/4/1142/0 | 3/0/189/0 | 1/2/188/1 |
| W4 128 | 5 | 122/96/1281/37 | 93/66/967/26 | 16/15/157/4 | 13/15/157/7 |

## Cost, review and limits

Scientific training cost:32000 added updates,2048000 examples,4096000 readouts,32768000 native steps. Evaluation cost:166 forwards,24064 cases,74624 readouts,596992 native steps. Fixed command exited0; `/usr/bin/time -p`: real848.30s, user668.52s, sys183.04s. No scientific retry.

Final checkpoint SHA256: float `8a58c04d6b0baed644075dcf4cd23accb8ef5406b2eb72d0f0152cf934cdea4f`; W4 `da7677541003cf7d260d8258cbe1f24a379d6ca07044e7042eea67fe7a464a4a`.

Independent review replayed all166 forwards with0 updates and matched complete evaluation dictionaries/predictions; replay time real78.96s/user75.31s/sys2.62s. Raw audit passed with0 forwards/updates (real1.38s). Review: `results/E33_CONTINUATION_REVIEW.md`; canonical report/time: `runs/e33_continuation/report.json`, `results/E33_SCIENCE_TIME.txt`.

Bounded QA ledger: `results/E33_QA_ACCOUNTING.json`. Final test run passed4/4 with known QA cost8 updates and2 eval forwards. Earlier passed QA has unavailable counts/path/timing; an interrupted QA attempt has unknown completion. Neither is scientific evidence. This is one adaptive seed0 diagnostic, not all-seed evidence, equal-budget h64 comparison, causal width claim, or proof of undertraining. No additional seed, extension or architecture change is authorized.
