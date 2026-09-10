"""CUDA-only PC inference suites after the frozen migration gate."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import torch

from looped_bitnet import register_e15 as dsl
from looped_bitnet import width_e32 as core
from scripts import followup_e37_e38 as followup
from scripts import length_wave_e35_e36 as wave
from scripts import pc_inference_migration as migration

ROOT = migration.ROOT
OUTPUT = ROOT / "runs" / "pc_inference_v1" / "science_cuda_v1"
LINEAGE = ROOT / "runs" / "e37_e38_science_v3" / "e38" / "runtime_lineage.json"
E37_PREFLIGHT = ROOT / "runs" / "e37_e38_preflight_v2"
OPS = ("ADD", "XOR", "SWAP")
STATES = list(dsl.STATE_ORDER)
STRATUM = {state: name for name, values in dsl.state_split().items() for state in values}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def signature(program: Sequence[str]) -> tuple[tuple[int, int], ...]:
    return dsl.semantic_signature(program)


def training_prefix_functions() -> set[tuple[tuple[int, int], ...]]:
    result = set()
    programs = set()
    streams = [core.stream(), *wave.build_streams()]
    for batches in streams:
        for batch in batches:
            for example in batch:
                programs.add(tuple(example.program))
    for program in programs:
        for size in range(1, len(program) + 1):
            result.add(signature(program[:size]))
    return result


def primary_functions() -> set[tuple[tuple[int, int], ...]]:
    manifest = migration.read_json(ROOT / "runs" / "e35_e36_preflight_v2" / "manifest.json")
    result = set()
    evaluation = manifest["evaluation"]["programs"]
    for length, values in evaluation.items():
        if int(length) not in (7, 8, 9, 10):
            continue
        for category in ("semantic_new",):
            for legality in ("allowed",):
                result.update(signature(program) for program in values[category][legality])
    return result


def composition_suite() -> dict[str, Any]:
    trained = training_prefix_functions()
    primary = primary_functions()
    excluded = trained | primary
    if len(trained) != 94 or len(primary) != 23:
        raise ValueError("accepted novelty function counts changed")
    result = {}
    for length in (12, 16, 24, 32):
        rng = random.Random(20260909 + length)
        seen_strings = set()
        seen_maps = set()
        rows = []
        draws = 0
        while draws < 4096 and len(seen_strings) < 1024 and len(rows) < 6:
            program = tuple(OPS[rng.randrange(3)] for _ in range(length))
            draws += 1
            if program in seen_strings:
                continue
            seen_strings.add(program)
            if any(a == "ADD" and b == "XOR" for a, b in zip(program, program[1:])):
                continue
            mapping = signature(program)
            if mapping in excluded or mapping in seen_maps:
                continue
            seen_maps.add(mapping)
            rows.append({"id": f"composition_L{length}_{len(rows)}", "program": list(program),
                         "length": length, "map": [list(state) for state in mapping]})
        if len(rows) != 6:
            raise ValueError(f"composition shortfall at length {length}")
        result[str(length)] = {"draws": draws, "distinct_candidates": len(seen_strings), "rows": rows}
    return {"seed": 20260909, "opcode_order": list(OPS),
            "prng": "random.Random(20260909 + length).randrange(3)",
            "excluded_training_prefix_function_count": len(trained),
            "excluded_primary_function_count": len(primary), "by_length": result,
            "selected_count": 24}


def padding_suite() -> dict[str, Any]:
    rows = []
    for prefix in (("ADD", "ADD"), ("XOR", "SWAP"), ("SWAP", "XOR")):
        q = prefix + ("SWAP",)
        for opcode in OPS:
            anchor = q + (opcode,)
            anchor_map = [dsl.execute_program(anchor, state) for state in STATES]
            for k in (0, 4, 6, 10, 14):
                program = q + ("SWAP", "SWAP") * k + (opcode,)
                mapping = [dsl.execute_program(program, state) for state in STATES]
                pre_o = [dsl.execute_program(q + ("SWAP", "SWAP") * k, state) for state in STATES]
                anchor_pre_o = [dsl.execute_program(q, state) for state in STATES]
                if mapping != anchor_map or pre_o != anchor_pre_o:
                    raise ValueError("padding exact-map identity failed")
                rows.append({"id": f"padding_{''.join(prefix)}_{opcode}_k{k}", "program": list(program),
                             "length": len(program), "prefix": list(prefix), "q": list(q),
                             "opcode": opcode, "k": k, "map": [list(state) for state in mapping]})
    return {"states": [list(state) for state in STATES], "program_count": len(rows), "rows": rows,
            "identity": "all k members equal the k0 map and pre-O state on all 256 states"}


def target_trace(program: Sequence[str], state: tuple[int, int]) -> list[list[int]]:
    return migration.oracle_trace(program, state)


def evaluate(model: torch.nn.Module, program: Sequence[str], device: torch.device) -> dict[str, Any]:
    x = torch.tensor([state[0] for state in STATES], dtype=torch.long, device=device)
    y = torch.tensor([state[1] for state in STATES], dtype=torch.long, device=device)
    ops = torch.tensor([[dsl.OP_TO_ID[op] for op in program] for _ in STATES], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits_x, logits_y = model(x, y, ops)
    decoded = torch.stack((logits_x.argmax(-1), logits_y.argmax(-1)), dim=-1).detach().cpu().tolist()
    predictions = []
    for index, state in enumerate(STATES):
        target = target_trace(program, state)
        predicted = decoded[index]
        correct = [predicted[pos] == target[pos] for pos in range(len(program))]
        predictions.append({"state": list(state), "stratum": STRATUM[state], "target_trace": target,
                            "predicted_trace": predicted, "prefix_joint_correct": correct,
                            "joint_final_correct": bool(correct[-1])})
    return summarize(program, predictions)


def summarize(program: Sequence[str], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    prefix = [sum(bool(item["prefix_joint_correct"][pos]) for item in predictions) for pos in range(len(program))]
    return {"program": list(program), "length": len(program), "states": len(predictions),
            "joint_final": sum(bool(item["joint_final_correct"]) for item in predictions),
            "final_x_correct": sum(item["predicted_trace"][-1][0] == item["target_trace"][-1][0] for item in predictions),
            "final_y_correct": sum(item["predicted_trace"][-1][1] == item["target_trace"][-1][1] for item in predictions),
            "full_trace": sum(all(item["prefix_joint_correct"]) for item in predictions),
            "prefix_joint": prefix, "predictions": predictions}


def paired_padding(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row["id"]): row for row in rows}
    result = []
    for row in rows:
        if int(row["k"]) == 0:
            continue
        anchor_id = str(row["id"]).rsplit("_k", 1)[0] + "_k0"
        current = by_id[anchor_id]
        counts = {key: 0 for key in ("short_correct_long_wrong", "short_wrong_long_correct", "both_correct", "both_wrong")}
        conditional = {"denominator": 0, "short_correct_long_wrong": 0, "short_wrong_long_correct": 0,
                       "both_correct": 0, "both_wrong": 0}
        left = {tuple(item["state"]): item for item in current["predictions"]}
        right = {tuple(item["state"]): item for item in row["predictions"]}
        for state in STATES:
            a, b = left[state], right[state]
            if a["target_trace"][-2:] != b["target_trace"][-2:]:
                raise ValueError("padding target pre-O identity failed")
            ac = bool(a["joint_final_correct"]); bc = bool(b["joint_final_correct"])
            key = "both_correct" if ac and bc else "short_correct_long_wrong" if ac else "short_wrong_long_correct" if bc else "both_wrong"
            counts[key] += 1
            af = all(a["prefix_joint_correct"]); bf = all(b["prefix_joint_correct"])
            key = "both_correct" if af and bf else "short_correct_long_wrong" if af else "short_wrong_long_correct" if bf else "both_wrong"
            conditional["denominator"] += 1 if all(a["prefix_joint_correct"][:-1]) and all(b["prefix_joint_correct"][:-1]) else 0
            conditional[key] += 1 if all(a["prefix_joint_correct"][:-1]) and all(b["prefix_joint_correct"][:-1]) else 0
        result.append({"anchor_id": anchor_id, "member_id": row["id"], "k": row["k"], "final": counts,
                       "conditional_final_O": conditional})
    return result


def pair_endpoints(endpoint_records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups = {}
    for record in endpoint_records:
        key = (record["arm"], record["seed"])
        groups.setdefault(key, {})[record["branch"]] = record
    result = []
    for (arm, seed), branches in sorted(groups.items()):
        if set(branches) != {"A", "B"}:
            continue
        for suite in ("padding", "compositions"):
            left = {str(row["id"]): row for row in branches["A"][suite]}
            right = {str(row["id"]): row for row in branches["B"][suite]}
            if set(left) != set(right):
                raise ValueError("A/B program IDs differ")
            counts = {key: 0 for key in ("A_correct_B_wrong", "A_wrong_B_correct", "both_correct", "both_wrong")}
            full = dict(counts)
            for identifier in sorted(left):
                lmap = {tuple(item["state"]): item for item in left[identifier]["predictions"]}
                rmap = {tuple(item["state"]): item for item in right[identifier]["predictions"]}
                for state in STATES:
                    a = bool(lmap[state]["joint_final_correct"]); b = bool(rmap[state]["joint_final_correct"])
                    counts["both_correct" if a and b else "A_correct_B_wrong" if a else "A_wrong_B_correct" if b else "both_wrong"] += 1
                    a = all(lmap[state]["prefix_joint_correct"]); b = all(rmap[state]["prefix_joint_correct"])
                    full["both_correct" if a and b else "A_correct_B_wrong" if a else "A_wrong_B_correct" if b else "both_wrong"] += 1
            result.append({"arm": arm, "seed": seed, "suite": suite, "program_count": len(left),
                           "final": counts, "full_trace": full})
    return result


def run() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable after migration gate")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    started = time.monotonic()
    archive_manifest, archive_checksums = migration.verify_prediction_archive()
    padding = padding_suite()
    compositions = composition_suite()
    proof = {"schema": "pc_inference_science_exact_maps_v1", "padding": padding,
             "compositions": compositions, "states": [list(state) for state in STATES]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "exact_map_proof.json", proof)
    manifest = {"schema": "pc_inference_science_cuda_manifest_v1", "status": "frozen",
                "archive_manifest_sha256": migration.sha256_file(migration.PREDICTIONS_MANIFEST),
                "archive_file_count": archive_manifest["file_count"], "archive_checksums_verified": len(archive_checksums),
                "checkpoint_labels": [item[0] for item in migration.ENDPOINTS], "endpoint_count": 12,
                "padding_program_count": 45, "composition_program_count": 24,
                "forward_budget": {"padding": 540, "compositions": 288, "total": 828},
                "device": torch.cuda.get_device_name(0), "dtype": str(torch.get_default_dtype())}
    write_json(OUTPUT / "manifest.json", manifest)
    progress = {"schema": "pc_inference_science_progress_v1", "status": "running", "attempted_forwards": 0,
                "completed_forwards": 0, "endpoint_index": 0, "suite": None}
    write_json(OUTPUT / "progress.json", progress)
    lineage = migration.read_json(LINEAGE)
    records = []
    with migration.windows_compatibility_adapter():
        for endpoint_index, (label, kind, arm, seed, branch) in enumerate(migration.ENDPOINTS, 1):
            checkpoint, _reference_path = migration.endpoint_paths(kind, arm, seed, branch)
            model, loaded_path, payload = migration.load_endpoint(kind, arm, seed, branch, lineage)
            if loaded_path != checkpoint or migration.sha256_file(checkpoint) != migration.manifest_hash_for_checkpoint(label):
                raise ValueError(f"checkpoint identity mismatch: {label}")
            model = model.to(torch.device("cuda"))
            model.eval()
            record = {"label": label, "kind": kind, "arm": arm, "seed": seed, "branch": branch,
                      "checkpoint": migration.posix_relative(checkpoint), "checkpoint_sha256": migration.sha256_file(checkpoint),
                      "model_digest": payload.get("model_digest"), "padding": [], "compositions": []}
            for suite_name, suite_rows in (("padding", padding["rows"]),
                                           ("compositions", [row for values in compositions["by_length"].values() for row in values["rows"]])):
                progress.update({"endpoint_index": endpoint_index, "suite": suite_name})
                write_json(OUTPUT / "progress.json", progress)
                destination = record[suite_name]
                for spec in suite_rows:
                    progress["attempted_forwards"] += 1
                    result = evaluate(model, spec["program"], torch.device("cuda"))
                    result.update({key: spec[key] for key in spec if key not in {"map"}})
                    destination.append(result)
                    progress["completed_forwards"] += 1
                    write_json(OUTPUT / "progress.json", progress)
            record["padding_pairs"] = paired_padding(record["padding"])
            write_json(OUTPUT / "endpoints" / f"{label.replace('/', '_')}.json", record)
            records.append(record)
    report = {"schema": "pc_inference_science_cuda_v1", "status": "complete", "device": torch.cuda.get_device_name(0),
              "dtype": str(torch.get_default_dtype()), "endpoint_count": len(records),
              "forwards": {"padding": 540, "compositions": 288, "total": 828},
              "endpoints": [{key: value for key, value in record.items() if key not in {"padding", "compositions"}}
                            | {"padding": record["padding"], "compositions": record["compositions"]} for record in records],
              "paired_A_vs_B": pair_endpoints(records), "elapsed_seconds": time.monotonic() - started}
    write_json(OUTPUT / "report.json", report)
    write_json(OUTPUT / "accounting.json", {"schema": "pc_inference_science_accounting_v1", "status": "complete",
                                             "attempted_forwards": 828, "completed_forwards": 828,
                                             "updates": 0, "endpoint_count": 12, "elapsed_seconds": report["elapsed_seconds"]})
    progress.update({"status": "complete", "attempted_forwards": 828, "completed_forwards": 828})
    write_json(OUTPUT / "progress.json", progress)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "endpoint_count": result["endpoint_count"],
                      "forwards": result["forwards"]}, indent=2))
