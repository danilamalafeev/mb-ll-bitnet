# E32 independent design review

2026-09-08. Reviewer: e32_review, Astra/low; independent of root design and Luna/xhigh implementation. Scope: `E32_WIDTH_PROTOCOL.md`, current orchestration/handoff/vision/log, historical E27/E24 protocols, and targeted constructor/model code inspection. No model forward, training, checkpoint replay or source edit performed.

**DESIGN CLEAR**. Root incorporated the following prefreeze clarifications; they are verified in the reviewed protocol:

- State the prospective exact parameter count: **335232** per h128 model. Of these, **331776** are weights at the 14 W4-eligible projections; **3456** remain outside them. In float all parameters use FP32; W4 also retains FP32 masters and merely fake-quantizes those projection weights on forward.
- Name L5 primary as **final-readout** accuracy. State the secondary fulltrace predicate explicitly: each program >=244/256 in all three seeds, with L4 and L5 predicates separately labeled. Secondary fulltrace does not replace primary final restoration.

Independent parameter derivation with width d and fixed FFN256: reader 4d²; four FFN up/down pairs 2048d; output heads 32d; bit input projections 8d; opcode embeddings 3d; role keys 2d; seven LayerNorms 14d. Total 4d²+2107d gives 151232 at d64 and 335232 at d128. W4 matrix inventory is 4d²+2080d; remaining parameters 27d. Actual constructor inventory must verify these before science.

Independent budget arithmetic: six models x16000 updates x64 examples and fixed mean two instruction readouts per training example gives 96000 updates, 6144000 examples, 12288000 readouts, 98304000 native8 substeps. Per model accepted seen/E21 evaluator contributes 71 forwards/8960 cases/23488 readouts; six L4 and six L5 programs contribute 12 forwards/3072 cases/13824 readouts. Thus six-model final evaluation is 498 forwards/72192 cases/223872 readouts/1790976 substeps. QA and independent replay remain separate.

Historical h64 comparison is acceptable conditional on frozen provenance checks: E24 continuation preserved complete optimizer/RNG and cumulative16000 recipe; E27 uses matched initial masters and actual float training-start RNG. Exact width64 factory reconstruction is a zero-forward check against those accepted initial states. Fresh h128 W4/float pair must share all named initial FP32 tensors and training-start RNG, with separate storage. Cross-width tensors are not identical; equal exposure is not equal FLOPs.

Preserving opcode coefficient1/8 is explicit and valid. Reader head dimension16->32 and its scaling, matrix capacity, h dimension, and FFN expansion ratio4->2 all change as part of this width package. A benefit cannot identify a hidden-state memory bottleneck. No reset or decoded feedback enters ordinary evaluation.

L5 threshold restoration, seen/E21/L4 prerequisites, and strict per-seed error reduction are correctly distinct. Signed W4-minus-float errors and width interaction are descriptive, with no assumed quantization disadvantage or significance claim. Frozen opened E28 functions and seeds constrain generalization claims.

Code CLEAR is still required before canonical preflight/science. This design review does not authorize unreviewed implementation, extra seeds, updates, model previews, or changes to historical artifacts.

Final reviewed protocol SHA256: `5cb304dec7f845e94b59754a0ad7db4a28054da3afa3c04cbf92b8b94ab33166`. Root may freeze this protocol and references; code CLEAR remains pending.
