# Late-memory causal check — 6 сентября 2026

## Результат

Inference-only check on 512 deterministic held-out cycle16 pairs. Each pair shares `s` and `s→b`, while `b→c_A` and `b→c_B` differ. The model starts from A, uses A's genuine K/V cache for steps 1–4, then B's genuine K/V cache for steps 5–8; no state reset or target feedback occurs.

- Pair manifest fingerprint: `d926b617ed3f301e5edb7991f76965688b990161af6f1b6769fc69e5d457a38d`; tables examined: 1164; all pair tables unique: `True`.
- Normal baseline accuracy: A→A `100.00%`; B→B `100.00%`.
- A-first4/B-last4 output matches: c_A `0.00%`, c_B `100.00%`, other `0.00%`.
- Conditional secondary result (both baselines correct): `512/512` c_B matches; excluded because a baseline was wrong: `0/512`.
- Reverse symmetry B-first4/A-last4 matches: c_A `100.00%`, c_B `0.00%`, other `0.00%`.
- h4 readout accuracy for b: A normal `100.00%`, B normal `100.00%`, A-first4/B-last4 `100.00%`.
- Full no-op A→A check: bitwise-equal batches `8/8`, allclose `True`, argmax equal `True`, max logit difference `0`.
- Runtime: `0.244` s (`0.476` ms/pair) on cpu.

## Interpretation and limits

If A-first4/B-last4 tracks c_B while the A→A no-op reproduces the normal model, this supports a causal dependence of the late output on B's late memory cache under this joint distribution. The intervention is a cache replacement after the h4 state has already been computed; it does not establish that h4 stores only b, and identical first edges do not imply identical h4 states because the full tables differ. A result here also does not establish general length generalization or a universal reasoning mechanism.

The conditional metric is secondary and reports its denominator explicitly; the primary output breakdown uses all pairs and never silently drops baseline failures. The h4 readout is a diagnostic shared output head, not target feedback.

## Verification

Independent final review completed with no blockers. The reviewer verified all 512 pairs / 1,024 unique held-out cycles, cache-switch indexing without a state reset, absence of target feedback, and agreement between the reported counts and artifacts. The full test suite passed: **31 passed**, including cache identity checks before and after step 4. This closes the authorized diagnostic stage; further experiments have not been launched.

## Artifacts

- Pair manifest: `/Users/danilamalafeev/Documents/ChatGPT/looped BITNET/results/LATE_MEMORY_CAUSAL_CHECK.pairs.json`
- Machine-readable report: `/Users/danilamalafeev/Documents/ChatGPT/looped BITNET/results/LATE_MEMORY_CAUSAL_CHECK.json`
- Checkpoint: `/Users/danilamalafeev/Documents/ChatGPT/looped BITNET/runs/intermediate_supervision_pilot/seed0/aux/checkpoint_best.pt`
