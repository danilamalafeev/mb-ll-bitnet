# Projected-value cycle screening result — 2026-09-11

The single registered baseline-vs-cycle-loss screening completed with exit code
0. A read-only saved-artifact audit passed and is recorded at
runs/pc_projected_value_cycle_screening_v1/screening/saved_audit.json.
The audit performed no model forward, backward, optimizer update or checkpoint
replay.

## Registered conditions

The ordinary arm used the accepted supervised latent-slot loss. The
cycle_loss arm used the same 64-update stream and added the detached
projected-V identity-cycle loss at weight 0.1.

| Measure | ordinary | cycle_loss | result |
|---|---:|---:|---|
| Final held-out weighted raw projected-V cycle loss | 0.0962584225 | 0.1105782790 | cycle arm is 14.876% worse; the registered 5% reduction fails |
| Final full-trace correctness | 0 / 512 | 0 / 512 | 0.0 percentage-point delta; the −5 pp guardrail passes |
| Final-step correctness (secondary) | 1 / 512 | 0 / 512 | recorded diagnostic; not the registered guardrail |

The protocol defines a result satisfying only one of the primary and
guardrail conditions as INCONCLUSIVE_ONE_CONDITION. Thus the overall
screening classification is inconclusive, while the primary cycle-loss
efficacy signal is negative: this short run provides no evidence that the
auxiliary loss improves the measured projected-V consistency. The guardrail
passes only because neither arm achieved a full correct trace on these 512
held-out cases.

## Execution integrity and cost

Both arms were independently initialized from the same accepted E36 endpoint,
with identical parent identity, adapter initialization and initial model,
optimizer and RNG identities. The embedded arm reports equal their standalone
reports. The source binding, launch-gate files, immutable input freeze,
stream prefix and both final checkpoint hashes match their saved declarations.

The registered counters were exact: 2 endpoint loads, 4 underlying
deserializations, 160 forwards, 10,240 cases, 62,976 readout positions,
503,808 native steps, 128 backwards, 128 optimizer updates, 2 committed
checkpoints, 2,450 training cycle windows and 106 detected cycle updates
(53 applied in the cycle arm). There were no runtime failures.

## Boundary

This is one seed, one endpoint, 64 fixed stream updates per arm and eight
predeclared programs over validation/test states. It is a bounded diagnostic,
not a superiority, causal or universal-correctness claim. The result does not
authorize a full pilot, tuning, retry, extra probes or a science run. Preserve
the raw reports, checkpoints and failed-attempt history; any continuation
requires a newly stated mechanism and a fresh source gate.
