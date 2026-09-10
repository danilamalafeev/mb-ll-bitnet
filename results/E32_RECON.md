# E32 bounded reconnaissance

2026-09-08. This note records mechanical compatibility checks before E32 code
CLEAR and scientific preflight. No E32 scientific training or L3/L4/L5 model
forward was run.

## Reuse and compatibility

The accepted E27 runner supplies the fixed 16,000-update stream, strict W4
operator inventory, AdamW update path, progress/checkpoint pattern, and
completed-versus-attempted forward counters. The E24/E22/E20 helpers supply the
historical FP32 optimizer configuration, loss, ordered data stream, seen plus
E21 evaluator, and strict saved-artifact conventions. E28's frozen selection
and scoring supply six fixed L4 plus six fixed L5 programs per model.

Historical h64 compatibility was checked with the new width-aware factory:
for seeds 0, 1, and 2, its W4 and float h64 initial named FP32 tensors match
the accepted E27/E25 masters exactly (151,232 parameters), and its recorded
training-start RNG matches the accepted historical RNG. The float arm is made
from the paired W4 masters by class-preserving replacement of the 14 projection
modules, so the two arms have identical tensors, ordering, shapes, dtypes,
disjoint storage, and training-start RNG within a seed.

The historical RNG source is explicit: seed0 uses the actual E20
`manual_seed(0)` training-start state; seeds1/2 use the saved accepted E22
initial RNG states. The h128 constructor draws generalized main modules after
`manual_seed(seed)` and draws each signed-bit input projection from a private
CPU generator seeded by the same nominal seed. Widths therefore have matched
initialization recipes, while tensor identity across widths is impossible and
is not claimed.

## h128 inventory

The constructor preserves four blocks, four heads, native8, opcode addition
coefficient 1/8, and FFN inner dimension256. Attention's existing scale changes
through `head_dim` (16 at h64, 32 at h128). The actual h128 parameter count is
335,232; the h64 control is 151,232. The h128 W4 arm has exactly the 14 E27
projection locations; W4 fake quantization is matrix absmax/7 with identity
STE and FP32 activations. The float arm uses matching FP32 linear modules.

## Fixed budgets and runtime estimate

Training is six sequential models in the frozen seed-major order
(float seed0, W4 seed0, float seed1, W4 seed1, float seed2, W4 seed2):
96,000 updates, 6,144,000 examples, 12,288,000 instruction readouts, and
98,304,000 native internal steps. Per new model's final evaluation is 83
forwards, 12,032 cases, 37,312 readouts, and 298,496 internal steps; all six
are 498, 72,192, 223,872, and 1,790,976 respectively.

The h64 E27 scientific run cost 1,570.99 seconds for three W4 models and the
h64 E26 weight-only run cost 751.74 seconds for three ternary models. The h128
FFN256 matrix work is approximately 2.22 times h64 for the dominant linear
terms, so a rough planning estimate for six h128 arms is several thousand CPU
seconds (about 55–120 minutes on this host, depending on arm and load). This is
an estimate from prior recorded costs, not a benchmark and not scientific
evidence.

The new implementation is confined to `looped_bitnet/width_e32.py`,
`scripts/width_e32.py`, and `tests/test_width_e32.py`; old experiments and
guards remain read-only.
