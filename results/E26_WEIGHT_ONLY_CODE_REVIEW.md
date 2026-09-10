# E26 independent code review — CLEAR

2026-09-08. Bounded independent review before canonical preflight or scientific training. Protocol and dual-comparator scheme applicable; no blocking findings.

- The only forward change from original BitLinear is `int8_activation(value)` to `value.float()`. The original ternary quantizer, identity weight STE, bias and FP32/autocast handling remain. All 14 existing layers are reclassified without parameter construction; exact initial tensors, independent historical RNG digests, 151232 parameters and native8 inventory are checked.
- Manifest reconstruction freezes source/protocol/comparator hashes, exact initial state/RNG, stream/target order, all eight repeats, costs, evaluator scopes and strata. All 261 protected files and 14 independently frozen comparator references match. E25/E24 canonical bytes remain intact.
- Scientific CLI fixes all three seeds and 16000 updates, original E20 update/AdamW/clipping and accepted evaluator. False scientific predicates do not stop subsequent seeds. Technical/nonfinite errors suspend with completed/attempted workload and preserved artifacts. No scientific retries or extra reload-update work occur.
- Strict loader rejects wrong types, schema, mode, seed/update, state inventory/dtype, model and optimizer/RNG digests, optimizer membership/step/moments, manifest and initial provenance. Genuine tiny QA compares the next model update, optimizer and RNG after reload.
- Both E25 and E24 comparisons pair identical programs/states/targets (and composition strata), verify paired marginals and include counts/deltas; evaluation retains instruction traces and stratum metrics. Primary 244/256 and seen 32/32 or 31/32 thresholds remain, with control excluded and all-three combined restoration gate.
- Independent targeted suite: **11 passed in 14.59s**, fresh `runs/e26_weight_only_review_qa_tests_v1`. This includes real three-seed legal-seen model training/evaluation, activation identity and weight STE gradients, tamper rejection, threshold/control and dual pairing checks. Canonical preflight/run directories were absent after review.

QA only: 4 completed training updates (8 examples, 64 recurrent substeps), 6 reload identity updates (12 examples, 96 substeps), 3 evaluation forwards (6 cases, 48 substeps): 13 model forwards and 208 recurrent substeps total. One intentionally injected evaluation exception occurs before a forward; its one trained seed and suspended report are included. Projection-only checks are separate. Full accounting and exact reviewed hashes: `results/E26_WEIGHT_ONLY_REVIEW_QA.json`.

This gate authorizes the fixed canonical preflight/run under root coordination. It is not an experimental-result ACCEPT; final independent 213-forward replay and prediction/DSL/strata/paired/provenance verification remain required.

## Reviewed SHA256

- `looped_bitnet/weight_only_e26.py`: `c9581bab5b647997fea1daeca86c0b5f21c580477dc21b8eab05621ce2bf7d51`
- `scripts/weight_only_e26.py`: `1f04cfcbd73d1dfbce8357c4cc5bdbb9f5564dd777efb067c5064e7695e0ee91`
- `tests/test_weight_only_e26.py`: `88e7017dff1f35368c1b5bcebc9ac99ce7154e4015be81f7970e4472b7213bbd`
- `results/E26_WEIGHT_ONLY_PROTOCOL.md`: `385518100c780c5301e8801531710473efbacf89c6c9f628257fc9bc54570fcc`
- `results/E26_PROTECTED_HASHES.json`: `ae1a4d30d5b402297942b57f334976cd61e33ea90db8f1043f48a79a700060d9`
- `results/E26_COMPARATOR_REFERENCE.json`: `11bf3a2c448b0611d0247b4439b1101176ba1f3aaf33657eeac52af9c7d1815b`
