# E32 independent preliminary factory review

2026-09-08, e32_review Astra/low, separate from executor. **Core factory verified; full CODE CLEAR remains pending.** This is an intermediate slice, while runner/checkpoint/evaluation implementation continues.

Independent `python -m pytest -q tests/test_width_e32.py`: 10 passed in2.17s. These tests execute no model forwards or updates. Additional independent zero-forward audit compared all named h64 factory tensors directly with frozen E18/E22 initial checkpoint tensors for seeds0/1/2 via torch.equal, rather than only using the historical factory: all exact. Each h128 float/W4 pair has equal named tensors and RNG with disjoint storage. Actual count335232, W4 projection weights331776, other parameters3456, all three seeds.

Source inspection confirms signed-bit ordering and FP32 projection, constructor/init ordering generalized from historical model,4 blocks/heads, FFN256, native8, opcode coefficient1/8, inherited W4 operator at14 locations, and no reset or decoded feedback. `forward` creates cache once and `step` carries h and the fixed initial KV between instructions. This no-reset conclusion is a source audit, not yet a runtime forward test. Head dimensions16/32 follow width. Historical training-start RNG is explicitly stored; runner must restore it before updates.

Required corrections before complete CODE CLEAR, sent directly to executor:

1. Current LABELS and RECON describe arm-major order float0/1/2 thenW40/1/2, whereas frozen protocol specifies seed-major float0,W40,float1,W41,float2,W42. Align manifest labels and actual runner order to protocol.
2. Reject boolean/non-integer seed metadata explicitly; Python True currently aliases seed1. Extend inventory guards to verify head count/head dimension, FFN/block shapes and width attributes rather than relying only on total count.
3. Draft checkpoint functions are outside this slice clearance. Full review must enforce raw tensor order/shape/dtype before load casts; exact optimizer groups/membership and moments shape/dtype; exact tensor step, not int(step); typed metadata; no fallback initial/source/stream provenance. Runner/manifest must freeze exact required source and reference inventory.

Costs for this review: 10 zero-forward tests plus zero-forward initial tensor audit;0 training updates,0 model forwards. No canonical E32 run created. Old code/artifacts untouched.

Inspected slice SHA256 looped_bitnet/width_e32.py: `ecf909fd2f5d4de8077f29559ec494036efbefb296f07f882bff9a99327c5e71`.

Inspected slice SHA256 tests/test_width_e32.py: `3b58d528f18cbd52b7a5cf2d078668dd73ba50f102c8ac02cedd8649696d0606`.
