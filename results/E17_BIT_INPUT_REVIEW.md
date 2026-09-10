# E17 independent result review — ACCEPT

Reviewer: Astra/low, 2026-09-07. The registered paired seed0 pilot is complete
and accepted. No additional training or expanded evaluation was performed.

## Verified findings

Bits versus learned, fixed update2000 on the same float core:

| Outcome | Learned | Bits | Bits−learned |
|---|---:|---:|---:|
| Length3 exposed-train joint macro | 63.5665% | 92.1131% | +28.5466pp |
| Length3 validation joint macro | 34.6726% | 80.6548% | +45.9821pp |
| ADD validation | 9/32 | 29/32 | +20 states |
| XOR validation | 24/32 | 31/32 | +7 states |
| SWAP validation | 32/32 | 32/32 | 0 states |

Bits primitive train is192/192 for each opcode; learned is191/192,192/192,
192/192 respectively. The remaining ADD29/32 and XOR31/32 validation errors
mean the original perfect32/32 primitive gate is NOT met. This pilot authorizes
no new compositions, reserved test cases or deeper evaluation.

## Independent verification

- Strict-loaded both final checkpoints in their actual learned/bits encoder
  modes, with the frozen E17 source/config/protocol, optimizer, stream/target,
  initial/common, arm/update and model-state provenance checks.
- Reconstructed seed0 initial models and validated both persisted initial
  states and their shared-core digest. All16 signed codes and both raw
  projection-weight hashes match `E17_BIT_ENCODING_REFERENCE.json`.
- Recomputed fixed stream/target digests and the complete coverage structure;
  run manifest and report agree. All6144 training program/state pairs exposed.
- Reloaded models in eval mode and reproduced ALL128 arm/program/split rows,
  including predictions, prefix correctness, final x/y/joint and full trace;
  recomputed all macros, gaps, display rows, paired outcomes and signed deltas
  exactly against the machine report.
- Separately recounted integer metrics from predicted traces with a direct
  ADD-mod16/XOR/SWAP interpreter, independently of aggregation helpers; every
  target, prefix/final/full-trace count and length macro agreed. Independently
  counted both/bits-only/learned-only/neither from final predictions; paired
  denominators and marginal-difference identities agreed in every row.
- Compared all32 E17 learned final state tensors with the existing E16 float
  final checkpoint: every tensor is bitwise identical. This establishes exact
  baseline reproduction, not an independent seed replication.
- Both progress/completion files match report records, have exactly eight
  finite observations at updates250..2000, and point to verified checkpoint
  byte hashes. Actual registered total:4000 optimizer updates,256000 examples,
  2048000 recurrent training substeps. Recorded run wall time27.0439s.
- Evaluation is exactly seen32 x train192/validation32 for both arms:14336
  program/state evaluations,36736 readout positions,146944 recurrent substeps.
  No reserved test32, unseen programs or lengths4/6 entered this replay.
  Reviewer replay is additional verification inference, not training.
- All118 protected file hashes remain unchanged. Final E17 source hashes match
  preflight and the hashes cleared in `E17_BIT_INPUT_CODE_REVIEW_FINAL.md`.

Review scripts ran successfully from `/tmp/e17_result_review.py` and
`/tmp/e17_counts_review.py`. They read existing artifacts; no old artifact or
source was changed. Primary machine evidence is
`runs/e17_bit_input/report.json` and its two strict-validated final checkpoints.

## Interpretation and stop

The signed-bit input package substantially improves both exposed train fit and
validation behavior here, while retaining the same float core, loss, data and
four substeps. Input functions/geometry were not matched at initialization;
only their marginal scale and common core were matched. The bits arm also has
1536 fewer trainable parameters (151232 versus152768) and a different training
constraint. This identifies a local representation/optimization-package effect,
not its sole mechanism or the only remaining bottleneck. One seed and one
budget cannot establish robustness; validation has informed earlier research.
It does not establish general recurrent reasoning or arbitrary-depth transfer.

The preregistered stop has been reached: one paired pilot plus independent
review. A four-versus-eight-substep comparison on the float bit-input core is
only a candidate requiring a separate prospective protocol; this review does
not implement or launch it, and no automatic extension is authorized.
