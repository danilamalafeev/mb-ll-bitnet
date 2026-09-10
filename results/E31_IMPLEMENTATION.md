# E31 implementation

Implemented `scripts/decoded_reset_e31.py` and targeted tests. Reads frozen E28 saved traces through E30 saved() and E28 strict validation/loaders; uses E30 evaluate() to preserve all module modes and external RNG in addition to model/optimizer/Torch RNG. Each of six L5 programs gets its own batch of256 OWN decoded fourth states, including duplicates. Scoring retains original identity and target, local target, oracle comparison, paired strata and hybrid full-trace gates. Direct/lookup disagreement is retained as a false control flag and does not replace direct output or suspend a legitimate numerical result.

17 targeted tests passed after review fix (8.61 seconds): own input vs oracle, duplicates, local vs original target, malformed saved/fresh data, original case scope, fourth vs prefix strata, empty denominator,244 gates, failure artifact/accounting, overwrite refusal, frozen preflight and retained lookup disagreement. Synthetic tests do not forward scientific checkpoints.

Authorized tiny seen ADD QA: `runs/e31_qa_seen_attempt1`; complete,6 attempted/completed forwards,12 cases/readouts,96 internal steps,0updates; `/usr/bin/time -p` elapsed4.59 seconds. All6 cells confirmed unchanged model/optimizer/Torch RNG, all module modes restored, external Python RNG unchanged, protected hashes unchanged.

QA manifest records the source version before two reporting-only changes: per-cell late-error predicate and reviewer-requested retained direct/lookup control. The final targeted tests cover both. QA retained unchanged as audit evidence; no extra QA forwards, scientific preflight or scientific evaluation run by executor. Final code review/freeze and scientific gate belong to root.
