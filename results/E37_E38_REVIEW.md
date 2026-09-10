# E37/E38 independent final review

**ACCEPT scientific artifacts. E38 registered replication criterion PASS.** One saved raw audit and one strict replay completed successfully; no retraining or replay retry. Human-readable report/accounting corrections below are now closed; final documents ACCEPT. Scientific artifact validity and the numerical result are distinct conclusions.

## Verified evidence

`runs/e37_e38_review_v3/status.json` records ACCEPT; `raw.json` PASS in10.417973333seconds; `replay.json` PASS in615.777502167seconds. Review command elapsed627.877809833seconds is separately recorded, not the science wall time. The reviewer used `results/e37_e38_review.py` (SHA256 `47afa8cfcae9f088b5e337bc92d29994ec463092946d2d13db34f907c91499f9`).

Independent DSL target reconstruction and exact state/program coverage passed for every saved row. The audit checks E37 function-preserving padding and within/cross-checkpoint pairs; E38 frozen common pools/probes, initial-state strata, full-trace/final metrics, parent scope/gates, both-budget paired directional predicates, hashes and parent lineage. E38 base and E37 output manifests remain byte-identical to frozen v2. Separate runtime lineage has the four exact own-parent registrations. Source/reference and accepted input hashes passed.

All22endpoints replay exactly:6E37 accepted seed0 endpoints,4new E38 u32000 parents,12E38 A/B_match/B endpoints. All full evaluation dictionaries/predictions agree, every optimizer has32state entries at the correct absolute update, and each evaluation preserves model/optimizer/RNG/mode. E37 used unchanged strict E36 loader; E38 used the new strict loader with immutable base manifest, separate runtime lineage and source referencev3. No model was trained during review.

Measured replay accounting:3296forwards,786176cases,4673536readout positions,37388288native steps,22top-level checkpoint-loader calls and388instrumented `torch.load` calls including nested preparation/lineage loads. These are reviewer costs, separate from science. Zero optimizer updates were permitted. Prior QA cost remains28updates/56positions/448steps; exact uninstrumented historical preparation load totals remain unknown.

Scientific training totals128000new updates,8192000examples,19456000positions,155648000native steps. Scientific evaluation totals equal the replay scope above. B4572 is a snapshot from each B8000 trajectory, not additional training. Science supervisor interval7910seconds is command wall time; do not identify it with training-loop time alone.

## E38 numerical result

Full-trace errors on the same6144-case allowed semantic-heldout L7–10 pool:

| Model | A8000 | B4572, instruction-matched | B8000, update-matched |
|---|---:|---:|---:|
| float seed1 |4024|25|12|
| float seed2 |4124|10|9|
| W4 seed1 |4390|31|16|
| W4 seed2 |3538|18|12|

Every precision/seed passes BOTH registered strict directional comparisons independently. Accepted E36 seed0 also passed both, so the conjunction across seeds0/1/2 and both precisions holds. All sampled own-branch attainment probes have100%full-trace accuracy at every length; this exceeds99%but is not exhaustive training-set mastery. No parent was excluded based on its descriptive prerequisite accuracy.

This replicates the benefit of the longer-trace/composition training package on the SAME opened finite pool across additional initializations. The primary24strings represent23distinct functions; no new blind pool or arbitrary-chain discovery occurred. Length is not isolated from semantic/compositional/example diversity. A and B use the same inference execution budget for an identical evaluated program; the matched comparisons concern TRAINING instructions or TRAINING updates. No statistical precision-superiority, h64, arbitrary-depth/general-reasoning or beyond-L10 conclusion follows.

## E37 descriptive result

Each entry below is final/full-trace errors over2304cases at that length (9programs×256initial states). Each endpoint has9216total cases across lengths3/5/7/9.

| Endpoint | L3 | L5 | L7 | L9 |
|---|---:|---:|---:|---:|
| float A |0/0|13/13|599/645|1661/1858|
| float B4572 |0/0|0/0|0/0|73/73|
| float B8000 |0/0|0/0|0/0|56/60|
| W4 A |1/1|70/71|1264/1339|2093/2235|
| W4 B4572 |0/0|0/0|1/1|207/217|
| W4 B8000 |0/0|0/0|1/1|155/161|

The long-training endpoints substantially reduce degradation under the fixed SWAP-pair padding, while nonzero L9 errors remain. This is descriptive evidence across fixed equivalent functions. Padding changes inference history composition and number of inference steps jointly; it does not isolate hidden-state memory versus cache/reader or establish arbitrary-chain execution.

## Provisional report corrections and final stop

The first assembled `results/E37_E38_RESULTS.md` had transcription/scope errors, independent of the scientific artifacts: E37 cell denominator256instead of2304; swapped float B4572/B8000 direct L9 entries; omitted W4 B4572 L7 error1/1; inaccurate inference-compute A/B confound. Its paired E37 table was already correct and must not be swapped. `E37_E38_ACCOUNTING.json` branch-evaluation step total33490096must be33540096 (=4192512×8). Executor was sent these precise corrections plus reviewed status/accounting updates. Parent L4/L5 table values and E38 primary/attainment values were checked against saved accepted outputs.

No further scientific computation is required or authorized. After the corrected human-readable report/accounting is checked, the fixed E37/E38 stop condition is complete: STOP and discuss with user. No new experiment, architecture change, extra seed/update, automatic routing or model waiting loop.


## Final document ACCEPT

The corrected final report and accounting are ACCEPT. Reviewed E37 denominator/direct rows/paired rows, branch-evaluation step total and training-versus-inference scope language. Reviewer made only final literal documentation corrections: raw audit covered saved scientific outputs but executed0model forwards/loads/updates, so its coverage is now separated from actual reviewer compute; parent1536denominator is program/state cases; E38 both-wrong counts are located in canonical primary_pairs rather than the summary accounting JSON. No scientific artifact, source, checkpoint, replay or model computation changed.

Final `E37_E38_RESULTS.md` SHA256 `3796291a78582541b2247166d7033c9a938650c8388252c922ec2054f8a283ca`; `E37_E38_ACCOUNTING.json` SHA256 `1d0ea1df91dc7f697c92da5ece8877cec535bec56613bbe0dc83d970978482fd`. The registered STOP is reached; no required computation or report correction remains.
