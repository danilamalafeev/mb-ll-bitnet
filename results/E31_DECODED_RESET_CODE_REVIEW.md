# E31 independent code review — CLEAR

2026-09-08. Independent reviewer, before canonical scientific execution. No scientific forwards executed by reviewer.

Reviewed frozen protocol/reference/protection, E31 runner/tests, inherited E30 validation/evaluation and E28 saved-schema/scoring/loader/evaluation. All 17 root-reference and 338 protected hashes matched. Own decoded predicted_trace[3] drives direct fresh one-op inputs. Original program/state identities and train/validation/test strata remain validated; duplicate decoded numbers remain separate cases. Local one-op correctness, original-final correctness, fourth-readout correctness, all-prefix4 correctness and hybrid fulltrace are distinct. Paired counts, empty error denominators and descriptive gates are appropriate. Aggregate zero-failure late-rescue predicate is equivalent to its per-cell conjunction; per-cell predicates are also saved.

One blocker was corrected before CLEAR: direct/E30 lookup mismatch previously raised a technical exception. Final code retains the actual prediction, records false direct_lookup_match, and permits a completed negative diagnostic. Malformed input/output schema still fails closed.

Causality: QATRegisterModel.forward (looped_bitnet/register_e15.py:317-325) initializes from x/y and passes only op_ids[:,index] to each step. NativeStepRegisterModel.step (looped_bitnet/step_budget_e18.py:122-136) receives only current opcode. Therefore saved fourth output has no fifth-op access. init_cache (register_e15.py:291-298) builds fixed input-derived KV; step updates h/substeps, not an accumulating record memory. Re-encoding own decoded numbers rebuilds this input context.

Independent validation: 35 targeted E31 plus inherited E30 tests passed, pytest 14.63 s (shell wall 15.307 s). This includes synthetic failure/nonfinite/mutation accounting, heterogeneous mode restoration, original scope/stratum rejection, duplicate own inputs, target distinctions, empty denominators, overwrite refusal, frozen preflight and retained mismatch. Independent seen ADD QA runs/e31_review_qa_seen_1 completed: attempted=completed=6 forwards,12 cases/readouts,96 internal steps,0 updates; shell wall 6.846 s. All model/optimizer/Torch RNG, external Python RNG and module-mode checks passed; protected files unchanged. Test and QA clocks include some concurrent CPU execution and are recorded separately.

Final reviewed hashes:
- `scripts/decoded_reset_e31.py`: `4fb7ea247f21801c658c1fd44aadf37fa147369ee7452cd6f8562f41015f7bcf`
- `tests/test_decoded_reset_e31.py`: `16ddb51a24c4fa954b3ec254a4822a54a4943fdbbce3bb000e76739652d69d2e`

CLEAR authorizes only frozen canonical preflight and fixed E31 scope. Scientific conclusions require independent saved-artifact recount and fixed-scope checkpoint replay.
