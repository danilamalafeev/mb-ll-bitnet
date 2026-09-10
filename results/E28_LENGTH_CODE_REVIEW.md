# E28 independent code review — CLEAR

Astra/low reviewer, 2026-09-08. Fixed inference-only scope is ready for canonical preflight and one scientific evaluation. No scientific forwards or training performed by this review.

Independent symbolic enumeration reproduced all twelve selected programs, 3072 exact target traces, complete lexicographic state order, and novelty counts (length4: 39 eligible syntax / 37 classes / 27 shorter classes; length5: 93 / 86 / 64). All15 root reference hashes matched. Audit: `runs/e28_length_review_design/symbolic_audit.json`.

Inspected accepted checkpoint loaders, native recurrent step and inherited forward/evaluate_program: one initial cache, continuous carried state, arbitrary instruction count, eight internal steps per instruction, no decoded-feedback/teacher forcing/reset. Loader inventory and frozen checkpoint hashes bind all six saved models. New runner preserves model/optimizer/RNG/mode, removes hooks in finally, records attempted/completed forward cost and preserves suspended partial outputs; output directories refuse reuse. Final and fulltrace decisions are separate perprogram conjunctions across both lengths/all three seeds. Raw prediction validation, independent DSL, strata alignment, first-divergence/recovery and paired final/fulltrace counts match protocol. Guard extends previous protected inventory and current references.

Validation: 14 targeted tests passed (3.34 seconds), unique `runs/e28_review_pytest_attempt1`; includes malformed identity/trace/strata, 243/244 threshold, recovery, injected exception/nonfinite and suspension/overwrite. Independent real legal-seen QA: six saved models, ADD, two states each; complete, all unchanged. Actual attempted = completed: 6 forwards / 12 cases / 12 readouts / 96 internal steps, zero training. `runs/e28_review_seen_attempt1/report.json`. Synthetic failure tests are QA, not checkpoint scientific evaluation. Full hashes/accounting: `runs/e28_length_review_design/code_review_accounting.json`.

Scientific budget remains 72 forwards / 18432 cases / 82944 readouts / 663552 internal steps, plus separately accounted independent replay. Prior seen results must remain explicitly historical; they are frozen references and are not re-evaluated here. Final claims are limited to the twelve selected novel functions: length and composition content jointly change, so this is not an isolated causal effect of length or arbitrary-length generalization.

Reviewed hashes:

- `scripts/length_transfer_e28.py`: `bb31e595c5790cf11e6e5b1dc239dea21ae7c10fc42c01736ea3411f62ff500a`
- `tests/test_length_transfer_e28.py`: `c6347811900dd335bebc9d50583116c228ad61ed2e5cd1310c4683e834486f33`
- `results/E28_PROTECTED_HASHES.json`: `8866a26e62a42da1aa08399335012b7e295d3ba0853b29bd1174f3790f1ac1dc`
- `results/E28_LENGTH_PROTOCOL.md`: `40e473db0ee0e87221f21430507986911f62d8d73fa6acb94d60cffcc3603c19`
- `results/E28_SELECTION.json`: `d4b94458e2dc7522265f601f99d68cd3fcdaa6b0cdef504207cc534bf385dcb0`
- `results/E28_REFERENCE.json`: `44a4365475764e93c5657d1ec8a498bdd35d0e1d9db76dc4a460d8918f329681`
