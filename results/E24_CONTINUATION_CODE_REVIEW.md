# E24 independent code review — CLEAR

Reviewed 2026-09-07 by independent Astra/low reviewer. CLEAR applies only to the registered fixed E24 continuation and the exact source hashes below. No E24 scientific forward or update was performed during this review. Final scientific results still require independent replay/count/provenance review.

## Frozen reviewed sources

| File | SHA256 |
|---|---|
| looped_bitnet/continuation_e24.py | 88806b75cc96a168c6d4bdd7158427071c80265ed9438264c065fd48dd1d50b5 |
| scripts/continuation_e24.py | cf560fc255c653bab0931f7d343ca6aaa65b733792b4df99a5866ba11fafc100 |
| tests/test_continuation_e24.py | c502c705aab948f832d370dcc04372c1ec559670e73b81ca0b1e7fb441e0710e |
| results/E24_CONTINUATION_PROTOCOL.md | a7b0489f63d5c268ba022dc766bfc326c6b656e14f70b1b02bf26eed38ab3820 |

All 231 protected file hashes verified unchanged after the checks.

## Verified scope

- Independently loaded all three accepted u8000 parents with zero forwards: model and AdamW digests match; every optimizer step is 8000; current torch RNG is `torch.equal` to stored parent RNG after constructor/loader calls. CPU4 and deterministic algorithms are enabled.
- Static review confirms accepted E20 full stream is reused for exactly 8000 additional updates per seed, fixed final checkpoint only, unchanged accepted update/loss/optimizer path, all three seeds evaluated despite a completed negative predicate, and no intermediate scientific evaluation.
- Final reload checks model/optimizer/RNG digests, seed/update/schema, inventory, parent provenance, source/config/manifest metadata and cost counters. Evaluation checks mode and model/optimizer/RNG immutability.
- Paired comparison checks program/state/target/stratum identities, outcome marginals, wins and losses; primary six-row threshold is independently reconstructed at 244/256 and checked against inherited conjunction. Equivalent control is excluded. Seen primitive 32/32 versus 31/32 and composition 31/32 versus 30/32 boundaries verified separately without forwards.
- Technical exceptions suspend remaining seeds and preserve report/progress plus actual completed updates and completed forward counters. A completed negative predicate continues remaining seeds. Fresh output directories and checkpoint writes refuse overwrite.

## Independent execution

Command: `.venv/bin/pytest -q tests/test_continuation_e24.py --basetemp=/private/tmp/e24-independent-review-qa`

Result: **6 passed in 6.03s**. Actual tiny legal seen-only QA exercised three-seed restore/update/save/reload/evaluation/report; uninterrupted versus reloaded continuation matches model/optimizer/RNG digests; metadata rejection and overwrite refusal; negative predicate continuation; technical failure after one successful update; evaluation failure after one completed legal forward; paired and threshold fixtures.

Additional zero-forward checks used retained QA artifacts to verify actual checkpoint counters (u4 = 4 updates, 4 examples, 32 training substeps), and independently reject model-tensor, optimizer-moment and RNG tampering. The first metadata inspection command used an incorrect pytest directory name and failed before loading an artifact; the corrected path succeeded with no additional forward or update.

Reviewer-only QA cost: **31 successful updates, 31 examples, 248 training internal substeps; 7 evaluation forwards, 7 cases, 7 readouts, 56 evaluation internal substeps**. This includes the two extra updates used for reload identity. One additional injected update attempt fails before a forward; one injected evaluation exception follows an already counted forward. Metadata-only checks add zero model work. Earlier executor/failed-attempt costs are separate in `results/E24_ATTEMPT_ACCOUNTING.json` and are not replaced by these reviewer totals.

## Limits

Forward hooks count completed top-level forwards. A runtime exception inside an unfinished forward cannot supply an exact partial internal-substep count; attempted update and technical failure remain recorded, with completed work reported separately. QA uses the same model and accepted update but a private small legal seen-only stream, not primary/control scientific predictions. Scientific final checkpoint and prediction claims are not yet established by code clearance.

No remaining blocking finding. Root may freeze canonical E24 preflight and launch one registered fixed sequential continuation run; any change to reviewed source requires review of that change before launch.
