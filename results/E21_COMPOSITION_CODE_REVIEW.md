# E21 independent code review

## Prospective review — HOLD (2026-09-07)

Reviewer: independent Astra/low. Scope: prospective E21 protocol, then stable new evaluator and targeted tests; no training or E21 model predictions.

Protocol is coherent: six semantically novel programs, 256 ordered states each, per-program final joint correctness >=244/256 with conjunction; equivalent ADD/XOR/XOR control excluded. Seven forwards total, 1792 cases, 5120 readouts and 40960 native substeps; state strata are descriptive exposure strata. One seed and adaptive checkpoint selection limit the inference.

HOLD is a code-clearance status, not a protocol objection: evaluator/tests are not yet available for stable review. Final clearance requires strict frozen identity/provenance, symbolic novelty, seven-pass reporting, actual legal-SEEN toy runner/report QA, asymmetric counts and 244/243 boundary controls. No scientific inference is cleared by this provisional entry.

## FINAL — CLEAR for one registered evaluation (2026-09-07)

Stable reviewed SHA256:

- `scripts/composition_e21.py`: `ad3c3e89d79b57389b3d9a2f48a51543df409354a0d50dcd249f52aab7dc8948`
- `tests/test_composition_e21.py`: `39106309e8483e57ede268c86308c5a78d8fceb975759a14d00172c00f7d1fff`
- Frozen E20 u8000 checkpoint: `4507dd19d95b8a90f6ee1b487abc6493a26ba19700b784eaa2949dbfaa75c85d`

Independent verification: `.venv/bin/python -m pytest -q tests/test_composition_e21.py` — 8 passed in 44.33s. Reviewed strict raw tensor inventory followed by accepted E20 metadata/model-digest loader; protected195 verification; source/dependency and target freezing before prediction; frozen E15 split equality/disjointness; symbolic novelty/control equivalence; seven-forward hook; no optimizer step/training call in E21; pre/post weight/checkpoint checks; report storage from single-pass traces. Native8 accepted recurrence is reused unchanged; tiny legal ADD hook also confirms eight blocks and one boundary readout.

First-draft issues were corrected before stable handoff: frozen state strata equality, target comparison, imported dependency hashes, unused mode removal, shared report builder and toy accounting. The legal-SEEN toy exercises evaluation, scoring, report construction and JSON writing, with seven forwards/2816 readouts; it has no trained dynamics. Its reported22528 native substeps are nominal QA accounting, not executed recurrent blocks. The separate real untrained ADD check executes16 state-substeps.

Reviewer additionally exercised the production predicate/report through synthetic stored rows and14 toy forwards receiving ADD only:244 passes,243 fails, all six passing with a zero-correct control still passes, and independently enumerated every asymmetric train/validation/test metric matches (final x250/y246/joint244, prefixes252/244, full trace240 on all states). These synthetic primary labels are not real E21 model predictions. This closes the committed threshold test's duplicated-predicate limitation without expanding evaluator code. No E21 trained-model inference or training occurred in code review.

No unresolved scientific blocker. Clearance is specific to these hashes and the one seven-program protocol; the coordinator must freeze preflight before inference, present six rows/predicates and separate control (CLI itself prints status/path only), then obtain independent result replay. No extra checkpoint, seed, training, tuning or follow-up evaluation is cleared.
