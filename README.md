# MB-LL-BitNet

Research prototype for multi-block looped latent BitNet reasoning.

The project studies a small quantized recurrent interpreter with reusable blocks, latent state, and controlled pointer-chasing experiments. The repository contains the model and experiment code, focused tests, and lightweight result summaries. Large checkpoints, raw traces, and machine-local experiment runs are intentionally kept outside version control.

## Development

```powershell
python -m pytest -q
```

The code is research software. Experimental claims and their reproducibility artifacts are documented separately from the source; do not infer scientific conclusions from an unregistered local run.


## Research status

See [project vision](docs/PROJECT_VISION.md), [current handoff](docs/HANDOFF.md), and [research log](docs/RESEARCH_LOG.md). Agent/model policy is in [AGENTS.md](AGENTS.md).

The latest [pair-carry intervention](results/PC_LATENT_SLOTS_PAIR_CARRY_RESULTS.md) repaired all 36 observed long-padding failures without regressions on the fixed 4,608-case focus set. It is an external program-aware diagnostic, not a learned generalization result. The [protocol](results/PC_LATENT_SLOTS_PAIR_CARRY_PROTOCOL.md) defines its scope and limitations.

Historical documents retain original machine paths and artifact references. Checkpoints and raw `runs/` outputs are not bundled here; those references alone do not make this checkout fully reproducible.
