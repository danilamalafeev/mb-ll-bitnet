# E15 register interpreter implementation

Implementation preflight is complete for the bounded ADD/XOR/SWAP task. The
new code is confined to `looped_bitnet/register_e15.py`,
`scripts/register_interpreter_e15.py`, `tests/test_register_e15.py`, and the
new `runs/register_e15_preflight/` artifacts.

The QAT model has 152,768 trainable parameters and the float GRU control has
152,720. Both expose a streaming per-opcode API: the initial x/y embeddings and
the fixed QAT key/value cache are created once, four internal updates are made
per opcode, and only the model's own hidden state crosses instruction
boundaries. Future opcodes, targets, and predicted registers never enter the
forward path. The QAT reader uses the four cyclic FFN blocks and two separate
register heads; the GRU uses H=132, a bias-free 128→132 projection, and a
native GRUCell.

The final preflight manifest is [runs/register_e15_preflight/v3/manifest.json](../runs/register_e15_preflight/v3/manifest.json); earlier initial and v2 preflights are preserved beside it. It freezes the 192/32/32 state split, all 39 short
programs, the six primary novel transformations, the length-4/6 secondary
selection, the cyclic 1/2/3 schedule with counts 666/668/666, source hashes,
and the seed-0 paired-batch digest. Secondary forbidden length-4 has all 26
programs because that group contains fewer than 32 candidates; the other
groups contain 32 selected programs.

Targeted verification: `pytest -q tests/test_register_e15.py` passed 6 tests.
It covers semantic execution, split and secondary determinism, exact parameter
counts, gradients, prefix causality, checkpoint save/load/evaluation, and
positive/negative complete gate checks. No E15 neural training, checkpoint
production, or scientific test inference was run.

The runner defaults to preflight and refuses to overwrite non-empty output.
For the final frozen manifest, use `--preflight-dir runs/register_e15_preflight/v3`.
Training requires the explicit `--train-cleared` switch; eval-only requires a
frozen manifest and a metadata-matching selected checkpoint.
