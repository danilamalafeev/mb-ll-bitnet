# E34 independent review — ACCEPT

2026-09-08. Reviewer: Astra / low. Scope: `scripts/error_analysis_e34.py`, `results/E34_ERROR_ANALYSIS.json`, and the four saved E32/E33 reports named in that JSON. No producer rerun, model import, checkpoint load, forward or training; inputs unchanged.

Independent standard-library reconstruction read all 12,288 saved cases (four reports × 12 programs × 256 states). Every reconstructed summary exactly matched E34, including per-program results, opcode/position numerators and denominators, register mismatch categories and final recovery counts. The audit also independently matched all correctness groups, first-divergence transition cells and all 12 displayed example records.

Verified:

- Input byte sizes and SHA-256 values match the E34 snapshot; all four reports are complete.
- Each report has six distinct L4 and six distinct L5 programs. Every program covers all 256 unique states with 192/32/32 train/validation/test cases.
- Every target trace matches independent ADD modulo 16, XOR and SWAP execution. Predicted traces have valid shape/range; stored prefix/final correctness matches direct comparison.
- Parent/final joins use program and initial-state keys; joined targets and strata agree. Raw L5 errors match report counters and E34 baselines: float 146 → 166; W4 159 → 133.
- Opcode eligibility requires every earlier decoded state to be correct, including the empty prefix at position 1. Numerators count first divergence, not all subsequent wrong readouts.
- Every illustrative trace, initial state, label, divergence position and correctness group agrees with its keyed source record. Examples are deterministic illustrations, not a representative statistical sample.

Critical interpretation checks:

| L5 measure, 1,536 cases per arm/stage | Float parent → final | W4 parent → final |
|---|---:|---:|
| First divergence at position 4 | 30 → 35 | 27 → 17 |
| First divergence at position 5 | 121 → 137 | 138 → 119 |
| Final correct after earlier error | 5 → 6 | 6 → 3 |
| Position-5 XOR first errors / eligible | 52/255 → 37/254 | 28/256 → 14/256 |
| Position-5 SWAP first errors / eligible | 69/1251 → 100/1247 | 110/1253 → 105/1263 |

No first divergence occurs in the first three decoded positions. This localizes observed failures late in these saved traces; it does not identify a latent-state cause. XOR/SWAP figures involve different programs and are descriptive, not causal opcode comparisons.

Top-level histograms and correctness groups pool **L4 + L5, 3,072 cases**, whereas the table above is L5 only. Pooled groups (persistent correct / recovered / introduced / persistent error) are float 2811 / 90 / 108 / 63 and W4 2800 / 132 / 102 / 38. Do not read those pooled deltas as L5-only deltas.

No blocking defect found. Acceptance covers the saved descriptive analysis, not an architectural/generalization claim or causal explanation. One seed and this fixed program set remain the evidence boundary. Audit completed successfully in one raw-data execution (about 0.33 seconds); no new model computation.

The human report `E34_ERROR_ANALYSIS.md` was also checked: L5-only correctness groups independently reproduce 1287/83/103/63 (float) and 1281/122/96/37 (W4), in persistent-correct/recovered/introduced/persistent-error order. Final L5 register mismatches reproduce 49/104/13 and 15/107/11. Every state in its ADD–SWAP–ADD–XOR–SWAP example at initial (2,3) matches the four raw reports. The main tables and restrained descriptive interpretation are consistent with the audit.
