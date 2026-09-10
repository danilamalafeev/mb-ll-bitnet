"""Independent stdlib-only saved trace audit; no model/checkpoint loads."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def truth(program, state):
    x, y = state
    out = []
    for op in program:
        if op == "ADD": x = (x + y) % 16
        elif op == "XOR": x ^= y
        elif op == "SWAP": x, y = y, x
        else: raise ValueError(op)
        out.append([x, y])
    return out


def flags(pred, target):
    errors = [i + 1 for i, (a, b) in enumerate(zip(pred, target)) if a != b]
    return not errors, pred[-1] == target[-1], errors


def paired(candidate, reference):
    if candidate and not reference: return "repair"
    if reference and not candidate: return "regression"
    return "both_correct" if candidate else "both_wrong"


def audit(out, saved):
    allstates = [(x,y) for x in range(16) for y in range(16)]
    ranked = sorted(allstates, key=lambda s:hashlib.sha256(f"E15-state-v1:{s[0]}:{s[1]}".encode()).hexdigest())
    strata = {s:("train" if i<192 else "validation" if i<224 else "test") for i,s in enumerate(ranked)}
    old = {}
    for f in (saved / "program_rows").glob("*.json"):
        row = json.loads(f.read_text(encoding="utf-8"))
        if row.get("detailed"): old[row["id"]] = row
    expected = {}
    for name, prefix in [("ADDADD",["ADD","ADD"]),("XORSWAP",["XOR","SWAP"]),("SWAPXOR",["SWAP","XOR"])]:
        for suffix in ("ADD","XOR","SWAP"):
            for k in (10,14):
                expected[f"padding_{name}_{suffix}_k{k}"] = prefix + ["SWAP"]*(2*k+1) + [suffix]
    assert set(old)==set(expected), "saved focus coverage"
    rows = {}
    for f in (out/"program_rows").glob("*.json"):
        row = json.loads(f.read_text(encoding="utf-8"))
        key = (row["arm"],row["id"])
        assert key not in rows, "duplicate output"
        rows[key]=row
    assert set(rows)=={(a,i) for a in ("sham","pair_carry") for i in expected}, "output coverage"
    groups = defaultdict(Counter)
    failures = []
    positions = 0
    for ident, program in expected.items():
        def indexed(row):
            assert row["program"]==program, "program mismatch"
            preds=row["predictions"]
            assert len(preds)==256, "state count"
            result={tuple(r["state"]):r for r in preds}
            assert len(result)==256 and set(result)==set(allstates), "state coverage"
            return result
        base=indexed(old[ident]); arms={a:indexed(rows[a,ident]) for a in ("sham","pair_carry")}
        for state in allstates:
            target=truth(program,state)
            data={}
            for arm in arms:
                r=arms[arm][state]; pred=r["predicted_trace"]
                assert r["stratum"]==strata[state], "stratum"
                assert r["target_trace"]==target, "target"
                assert len(pred)==len(target) and all(len(v)==2 and all(type(x) is int and 0<=x<16 for x in v) for v in pred), "decoded shape"
                data[arm]=flags(pred,target)
                positions+=len(pred)
            assert arms["sham"][state]["predicted_trace"]==base[state]["predicted_trace"], "sham must match accepted traces"
            af,ae,ax=data["sham"]; bf,be,bx=data["pair_carry"]
            keys=["all",f"length:{len(program)}",f"stratum:{strata[state]}",f"program:{ident}","originally_correct" if af else "originally_failed"]
            if strata[state]!="train": keys.append("heldout")
            for key in keys:
                g=groups[key];g["cases"]+=1;g["full_"+paired(bf,af)]+=1;g["final_"+paired(be,ae)]+=1
                for arm,(full,final,errors) in data.items():
                    g[arm+"_full_correct"]+=int(full);g[arm+"_final_correct"]+=int(final)
                    if errors:g[arm+"_first_error_"+str(errors[0])]+=1
                    g[arm+"_padding_wrong_positions"]+=sum(3<=i<len(program) for i in errors)
                    g[arm+"_suffix_wrong_positions"]+=sum(i==len(program) for i in errors)
                    g[arm+"_prefix_wrong_positions"]+=sum(i<=2 for i in errors)
            if not af or not bf:failures.append({"id":ident,"state":list(state),"sham_errors":ax,"pair_carry_errors":bx})
    assert positions==258048, "position budget"
    return {"status":"ACCEPT_TRACES", "model_calls":0,"deserializations":0,"program_forwards_audited":36,"cases_audited":9216,"positions_audited":positions,"native_steps_expected":positions*8,"aggregates":dict(groups),"changed_or_failed_cases":failures,"positive_pilot":groups["all"]["full_repair"]==36 and groups["all"]["full_regression"]==0,"limitations":["opened single endpoint/pool","external program-aware pair-carry intervention","does not uniquely identify writer or establish learned generalization","accounting/source integrity checked separately"]}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);ap.add_argument("--saved",type=Path,required=True);ap.add_argument("--report",type=Path,required=True);a=ap.parse_args()
    assert truth(["SWAP","SWAP"],(3,7))==[[7,3],[3,7]]
    assert truth(["ADD","XOR"],(15,2))==[[1,2],[3,2]]
    assert paired(True,False)=="repair" and paired(False,True)=="regression"
    assert paired(True,True)=="both_correct" and paired(False,False)=="both_wrong"
    r=audit(a.out,a.saved)
    with a.report.open("x",encoding="utf-8") as f:json.dump(r,f,indent=2)
    print(json.dumps({k:r[k] for k in ("status","positive_pilot","program_forwards_audited","positions_audited")}))

if __name__=="__main__":main()
