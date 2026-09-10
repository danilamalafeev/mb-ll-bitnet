# E32 implementation and QA record

2026-09-08. This record covers implementation and legal QA only. Scientific
E32 training and the fixed seen/E21/L4/L5 evaluation have not been run.

The implementation is in the new E32 module and runner. The factory preserves
the historical h64 initialization identity, creates h128 FFN256 models with
335,232 actual parameters, keeps the opcode coefficient at 1/8, and enforces
the 14-projection W4 inventory. The runner uses the frozen seed-major order
`float128_seed0`, `w4128_seed0`, `float128_seed1`, `w4128_seed1`,
`float128_seed2`, `w4128_seed2`. The evaluator checks the complete seen,
E21, and E28 scopes and records paired final-readout and full-trace outcomes.

Checkpoint loading validates typed identity, exact state key/order/CPU
FP32 tensors, complete AdamW parameter-group configuration, optimizer state
shapes/dtypes/steps, RNG payload, source/protected/reference provenance, and
manifest-derived metadata before accepting a payload. The QA reload check
keeps an independent deep-copied presave model/optimizer branch and compares
that branch with a separately loaded checkpoint branch after one update,
including optimizer state and Torch RNG.

The retained legal QA artifact is [runs/e32_width_qa](/Users/danilamalafeev/Documents/ChatGPT/looped%20BITNET/runs/e32_width_qa). It contains six model cells, two legal ADD updates per cell, finite two-state evaluation, strict checkpoint reload, and the reload identity controls. That artifact was produced before the reload-control accounting correction: its saved `qa_reload_next_update_cost` field says six logical controls, while the two actual branches per cell are twelve update executions. It remains unchanged for audit. The corrected runner and tests count both branches as 12 updates, 24 examples, 24 readouts, and 192 native substeps.

The latest targeted suite has 15 passing tests. It includes the retained-factory checks, legal runner QA, strict tamper rejection, synthetic saved-report assembly and threshold checks, paired-metric identity checks, and a technical-failure counter check. No scientific forward was used by the synthetic report tests.
