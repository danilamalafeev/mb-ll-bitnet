"""PC-only inference migration gate for the accepted E36/E38 endpoints.

This module is deliberately separate from the frozen experiment runners.  It
uses a process-local Windows path adapter, never rewrites accepted artifacts,
and stops before science if the saved decoded traces do not migrate exactly.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import itertools
import json
from pathlib import Path
import random
import time
from typing import Any, Iterator, Mapping, Sequence

import torch

from looped_bitnet import register_e15 as dsl
from looped_bitnet import width_e32 as core
from scripts import followup_e37_e38 as followup
from scripts import length_wave_e35_e36 as wave
from scripts import width_e32 as e32

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_ROOT = ROOT.parent / "pc_predictions_v1" / "project"
PREDICTIONS_MANIFEST = PREDICTIONS_ROOT / "PREDICTIONS_MANIFEST.json"
OUTPUT = ROOT / "runs" / "pc_inference_v1" / "migration_v1"
LINEAGE = ROOT / "runs" / "e37_e38_science_v3" / "e38" / "runtime_lineage.json"
E37_PREFLIGHT = ROOT / "runs" / "e37_e38_preflight_v2"
E36_PREFLIGHT = ROOT / "runs" / "e35_e36_preflight_v2"
E32_PREFLIGHT = ROOT / "runs" / "e32_width_preflight"
OPS = ("ADD", "XOR", "SWAP")
L7 = 7

ENDPOINTS = (
    ("float128_seed0/A", "e36", "float", 0, "A"),
    ("float128_seed0/B", "e36", "float", 0, "B"),
    ("w4128_seed0/A", "e36", "w4", 0, "A"),
    ("w4128_seed0/B", "e36", "w4", 0, "B"),
    ("float128_seed1/A", "e38", "float", 1, "A"),
    ("float128_seed1/B", "e38", "float", 1, "B"),
    ("w4128_seed1/A", "e38", "w4", 1, "A"),
    ("w4128_seed1/B", "e38", "w4", 1, "B"),
    ("float128_seed2/A", "e38", "float", 2, "A"),
    ("float128_seed2/B", "e38", "float", 2, "B"),
    ("w4128_seed2/A", "e38", "w4", 2, "A"),
    ("w4128_seed2/B", "e38", "w4", 2, "B"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def posix_relative(path: Path, root: Path = ROOT) -> str:
    path = Path(path)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


@contextmanager
def windows_compatibility_adapter() -> Iterator[None]:
    """Adapt only platform-sensitive path comparisons inside this process."""
    original_wave = wave._relative
    original_followup = followup._relative
    original_e32_loader = e32.load_manifest

    def frozen_e32_loader(path: Path = e32.e.PREFLIGHT, root: Path = ROOT) -> dict[str, Any]:
        path = Path(path)
        path = path if path.is_absolute() else Path(root) / path
        manifest_path = path / "manifest.json"
        expected = "e3e0fbc1a06ae9480db7119a0f405462b45634349ba76d0fa2cd18e49680b083"
        if sha256_file(manifest_path) != expected:
            raise ValueError("frozen E32 manifest bytes changed")
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError("frozen E32 manifest is not an object")
        return manifest

    wave._relative = posix_relative
    followup._relative = posix_relative
    e32.load_manifest = frozen_e32_loader
    try:
        yield
    finally:
        wave._relative = original_wave
        followup._relative = original_followup
        e32.load_manifest = original_e32_loader


def verify_prediction_archive() -> tuple[dict[str, Any], dict[str, str]]:
    manifest = read_json(PREDICTIONS_MANIFEST)
    if manifest.get("schema") != "pc_predictions_v1_manifest" or manifest.get("file_count") != 24:
        raise ValueError("prediction archive manifest identity changed")
    checksums: dict[str, str] = {}
    for item in manifest["files"]:
        relative = str(item["path"])
        path = PREDICTIONS_ROOT / relative
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            raise ValueError(f"prediction archive size mismatch: {relative}")
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise ValueError(f"prediction archive hash mismatch: {relative}")
        checksums[relative] = actual
    sums = PREDICTIONS_ROOT.parent / "SHA256SUMS"
    if not sums.is_file():
        raise ValueError("prediction archive SHA256SUMS missing")
    listed = {}
    for line in sums.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        listed[relative] = digest
    expected_manifest_sum = sha256_file(PREDICTIONS_MANIFEST)
    if listed.get("project/PREDICTIONS_MANIFEST.json") != expected_manifest_sum:
        raise ValueError("prediction archive manifest checksum mismatch")
    for relative, digest in checksums.items():
        if listed.get("project/" + relative) != digest:
            raise ValueError(f"SHA256SUMS mismatch: {relative}")
    return manifest, checksums


def oracle_trace(program: Sequence[str], state: tuple[int, int]) -> list[list[int]]:
    current = state
    trace = []
    for opcode in program:
        current = dsl.execute_program((opcode,), current)
        trace.append([int(current[0]), int(current[1])])
    return trace


def prove_exact_maps() -> dict[str, Any]:
    states = list(dsl.STATE_ORDER)
    padding_programs = []
    pre_o_identities = []
    for prefix in (("ADD", "ADD"), ("XOR", "SWAP"), ("SWAP", "XOR")):
        q = prefix + ("SWAP",)
        for opcode in OPS:
            anchor = q + (opcode,)
            anchor_map = [dsl.execute_program(anchor, state) for state in states]
            for k in (0, 4, 6, 10, 14):
                program = q + ("SWAP", "SWAP") * k + (opcode,)
                current_map = [dsl.execute_program(program, state) for state in states]
                if current_map != anchor_map:
                    raise ValueError("padding exact-map proof failed")
                before_o = [dsl.execute_program(q + ("SWAP", "SWAP") * k, state) for state in states]
                anchor_before_o = [dsl.execute_program(q, state) for state in states]
                if before_o != anchor_before_o:
                    raise ValueError("padding pre-O identity proof failed")
                padding_programs.append({"program": list(program), "length": len(program), "k": k, "opcode": opcode})
                pre_o_identities.append({"q": list(q), "k": k, "states": len(states), "equal": True})

    # The candidate generator is pure and deterministic.  Its stream order is
    # explicit so a later science runner cannot silently refill or reshuffle it.
    candidates_by_length = {}
    accepted = []
    for length in (12, 16, 24, 32):
        rng = random.Random(20260909 + length)
        distinct = []
        seen_strings = set()
        draws = 0
        while draws < 4096 and len(seen_strings) < 1024 and len(distinct) < 6:
            program = tuple(OPS[rng.randrange(len(OPS))] for _ in range(length))
            draws += 1
            if program in seen_strings:
                continue
            seen_strings.add(program)
            if any(a == "ADD" and b == "XOR" for a, b in zip(program, program[1:])):
                continue
            distinct.append(program)
        if len(distinct) < 6:
            raise ValueError(f"new-composition candidate shortfall at length {length}")
        rows = []
        for program in distinct:
            mapping = [list(dsl.execute_program(program, state)) for state in states]
            rows.append({"program": list(program), "length": length, "map": mapping})
        candidates_by_length[str(length)] = {"draws": draws, "distinct_candidates": len(seen_strings), "rows": rows}
        accepted.extend(rows)

    return {
        "schema": "pc_inference_exact_map_proof_v1",
        "states": [list(state) for state in states],
        "padding": {"program_count": len(padding_programs), "programs": padding_programs,
                     "pre_o_identity_count": len(pre_o_identities), "pre_o_identities": pre_o_identities},
        "new_compositions": {"seed": 20260909, "opcode_order": list(OPS),
                              "prng": "random.Random(20260909 + length).randrange(3)",
                              "by_length": candidates_by_length,
                              "selected_count": len(accepted)},
    }


def endpoint_paths(kind: str, arm: str, seed: int, branch: str) -> tuple[Path, Path]:
    if kind == "e36":
        checkpoint = ROOT / "runs" / "e35_e36_length_wave" / "science" / f"{arm}128_seed0" / branch / "u40000.pt"
        reference = PREDICTIONS_ROOT / "runs" / "e37_e38_science_v3" / "e37" / f"{arm}128_seed0" / branch / "predictions.json"
    else:
        checkpoint = ROOT / "runs" / "e37_e38_science_v3" / "e38" / "branches" / f"{arm}128_seed{seed}" / branch / "u40000.pt"
        reference = PREDICTIONS_ROOT / "runs" / "e37_e38_science_v3" / "e38" / "branches" / f"{arm}128_seed{seed}" / branch / "predictions.json"
    return checkpoint, reference


def choose_reference_row(reference: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    rows = reference.get("rows") or reference.get("e36", {}).get("rows")
    if not isinstance(rows, list):
        raise ValueError("reference rows missing")
    selected = [row for row in rows if int(row.get("length", -1)) == L7]
    if reference.get("e36"):
        selected = [row for row in selected if row.get("category") == "semantic_new" and row.get("legality") == "allowed"]
        rule = "lexicographically_first_existing_allowed_primary_L7"
    else:
        rule = "lexicographically_first_existing_E35_L7_anchor"
    if not selected:
        raise ValueError("reference L7 row missing")
    row = min(selected, key=lambda item: tuple(item["program"]))
    if int(row.get("states", -1)) != 256 or len(row.get("predictions", [])) != 256:
        raise ValueError("reference L7 state scope changed")
    return row, rule


def validate_reference_row(row: Mapping[str, Any]) -> tuple[tuple[str, ...], dict[tuple[int, int], Mapping[str, Any]]]:
    program = tuple(str(op) for op in row["program"])
    if len(program) != L7 or any(op not in OPS for op in program):
        raise ValueError("reference L7 program malformed")
    values = {}
    for item in row["predictions"]:
        state = tuple(int(value) for value in item["state"])
        if state in values or state not in dsl.STATE_ORDER:
            raise ValueError("reference state scope malformed")
        expected = oracle_trace(program, state)
        if item.get("target_trace") != expected:
            raise ValueError(f"DSL target mismatch for state {state}")
        values[state] = item
    if set(values) != set(dsl.STATE_ORDER):
        raise ValueError("reference does not cover all 256 states")
    return program, values


def load_endpoint(kind: str, arm: str, seed: int, branch: str, lineage: Mapping[str, Any]) -> tuple[torch.nn.Module, Path, dict[str, Any]]:
    checkpoint, _ = endpoint_paths(kind, arm, seed, branch)
    if kind == "e36":
        manifest = wave.load_manifest(E36_PREFLIGHT, root=ROOT)
        model, _optimizer, payload = wave.load_checkpoint(checkpoint, manifest, arm=arm, branch=branch,
                                                          expected_update=40000, root=ROOT)
    else:
        manifest = followup.load_manifest(E37_PREFLIGHT, root=ROOT)
        model, _optimizer, payload = followup._load_extended_checkpoint(
            checkpoint, manifest, arm=arm, seed=seed, stage="branch", branch=branch,
            expected_update=40000, qa=False, lineage=lineage, root=ROOT)
    model.eval()
    return model, checkpoint, payload


def run_model(model: torch.nn.Module, program: Sequence[str], device: torch.device) -> tuple[list[list[list[int]]], torch.Tensor, torch.Tensor]:
    model = model.to(device)
    states = list(dsl.STATE_ORDER)
    x = torch.tensor([state[0] for state in states], dtype=torch.long, device=device)
    y = torch.tensor([state[1] for state in states], dtype=torch.long, device=device)
    ops = torch.tensor([[dsl.OP_TO_ID[opcode] for opcode in program] for _ in states], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits_x, logits_y = model(x, y, ops)
    decoded = torch.stack((logits_x.argmax(-1), logits_y.argmax(-1)), dim=-1).detach().cpu().tolist()
    return decoded, logits_x.detach().cpu(), logits_y.detach().cpu()


def compare_decoded(decoded: list[list[list[int]]], references: Mapping[tuple[int, int], Mapping[str, Any]]) -> bool:
    for index, state in enumerate(dsl.STATE_ORDER):
        if decoded[index] != references[state]["predicted_trace"]:
            return False
    return True


def run() -> dict[str, Any]:
    started = time.monotonic()
    manifest, archive_checksums = verify_prediction_archive()
    proof = prove_exact_maps()
    lineage = read_json(LINEAGE)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    protected = "verified"
    try:
        core.verify_protected_hashes(ROOT)
    except Exception as exc:
        protected = f"failed: {type(exc).__name__}: {exc}"
    historical_missing = []
    reference_snapshot = read_json(ROOT / core.REFERENCE)
    for relative, expected in reference_snapshot["sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            historical_missing.append(relative)

    rows = []
    loaded = 0
    cpu_pass = True
    with windows_compatibility_adapter():
        for label, kind, arm, seed, branch in ENDPOINTS:
            checkpoint, reference_path = endpoint_paths(kind, arm, seed, branch)
            reference = read_json(reference_path)
            reference_row, selection_rule = choose_reference_row(reference)
            program, reference_by_state = validate_reference_row(reference_row)
            model, loaded_path, payload = load_endpoint(kind, arm, seed, branch, lineage)
            loaded += 1
            if loaded_path != checkpoint or sha256_file(checkpoint) != manifest_hash_for_checkpoint(label):
                raise ValueError(f"checkpoint identity mismatch: {label}")
            cpu_start = time.monotonic()
            cpu_decoded, cpu_x, cpu_y = run_model(model, program, torch.device("cpu"))
            cpu_seconds = time.monotonic() - cpu_start
            matches_reference = compare_decoded(cpu_decoded, reference_by_state)
            cpu_pass = cpu_pass and matches_reference
            item: dict[str, Any] = {
                "label": label, "kind": kind, "arm": arm, "seed": seed, "branch": branch,
                "checkpoint": posix_relative(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                "reference": posix_relative(reference_path, PREDICTIONS_ROOT.parent),
                "reference_sha256": sha256_file(reference_path), "selection_rule": selection_rule,
                "program": list(program), "model_digest": payload.get("model_digest"),
                "cpu": {"status": "pass" if matches_reference else "mismatch", "seconds": cpu_seconds,
                        "matches_saved_decoded_trace": matches_reference, "decoded_trace": cpu_decoded,
                        "target_trace": [reference_by_state[state]["target_trace"] for state in dsl.STATE_ORDER]},
            }
            if matches_reference and torch.cuda.is_available():
                cuda_start = time.monotonic()
                cuda_decoded, cuda_x, cuda_y = run_model(model, program, torch.device("cuda"))
                cuda_seconds = time.monotonic() - cuda_start
                abs_diff = torch.maximum((cpu_x - cuda_x).abs(), (cpu_y - cuda_y).abs()).max().item()
                rel_diff = torch.maximum((cpu_x - cuda_x).abs() / cpu_x.abs().clamp_min(1e-12),
                                          (cpu_y - cuda_y).abs() / cpu_y.abs().clamp_min(1e-12)).max().item()
                item["cuda"] = {"status": "pass" if cuda_decoded == cpu_decoded else "mismatch",
                                 "seconds": cuda_seconds, "matches_cpu_decoded_trace": cuda_decoded == cpu_decoded,
                                 "max_abs_logit_diff": abs_diff, "max_rel_logit_diff": rel_diff,
                                 "decoded_trace": cuda_decoded}
            else:
                item["cuda"] = {"status": "blocked_by_cpu_gate" if not matches_reference else "unavailable"}
            rows.append(item)
            if not matches_reference:
                break

    cuda_pass = bool(rows) and all(item.get("cuda", {}).get("status") == "pass" for item in rows)
    report = {
        "schema": "pc_inference_migration_v1",
        "status": "complete" if cpu_pass else "blocked_cpu_reference_mismatch",
        "science_authorized": bool(cpu_pass and cuda_pass),
        "archive": {"manifest": posix_relative(PREDICTIONS_MANIFEST, ROOT.parent),
                    "manifest_sha256": sha256_file(PREDICTIONS_MANIFEST), "files": manifest["file_count"],
                    "bytes": manifest["bytes"], "checksums_verified": len(archive_checksums)},
        "integrity": {"protected_snapshot": protected, "historical_reference_missing": historical_missing},
        "proof": {"path": posix_relative(OUTPUT / "exact_map_proof.json"),
                  "padding_programs": proof["padding"]["program_count"],
                  "new_composition_programs": proof["new_compositions"]["selected_count"]},
        "endpoint_count": len(ENDPOINTS), "loaded_endpoints": loaded,
        "forwards": {"cpu_attempted": len(rows), "cpu_completed": sum(item["cpu"]["status"] == "pass" for item in rows),
                      "cuda_attempted": sum("decoded_trace" in item.get("cuda", {}) for item in rows),
                      "cuda_completed": sum(item.get("cuda", {}).get("status") == "pass" for item in rows)},
        "endpoints": rows,
        "elapsed_seconds": time.monotonic() - started,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "exact_map_proof.json").write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def manifest_hash_for_checkpoint(label: str) -> str:
    mapping = {
        "float128_seed0/A": "1c78ee7e13b8c747171be6db4d56d5d9c1df5eaec26bcacf2aab3f7c24a18774",
        "float128_seed0/B": "59feda5e7d011c6c2a8ddb644a4dd9c74526a7af72cc657f0c927afe373dd3e9",
        "w4128_seed0/A": "3fcb61d6cf33fe7f8bb164f8660729a86cc39db960c42e8115d3bbfd0a730440",
        "w4128_seed0/B": "3ae5f6d4a60cb86bc88386c96c7a6683e306d0f6744e5072ff860dd5e63d2b45",
        "float128_seed1/A": "a837211d9c393f2aee36d6622176877190c9605c1fa36645559e64e23a1ecdc9",
        "float128_seed1/B": "39da90c43ddd78b8bd30b3f52f671743a11c7e7ccaae81f397032a12f7ba5458",
        "w4128_seed1/A": "08e0201de7b23c3938a64f46242b81a452b1c0ba91661c4b0b3a1fc105becadc",
        "w4128_seed1/B": "0d4e0045cb52f0b2c32afa6f0149e895b122977a3022901122f1ef8fd7ba13b5",
        "float128_seed2/A": "b5ac51159a9f1775cef14c13eda85756d44c23a5466d18300fd04bf781c3adfb",
        "float128_seed2/B": "b62a06be9310acae0c1e8d4a38bfa547d48de0811430d957bd488be35b05f4a8",
        "w4128_seed2/A": "e17d957b62ee18e09915679425b761936177eb0bd289acf4b014fe069e81b9e8",
        "w4128_seed2/B": "93945e847e7178965bf545a3e5dc61903ca8e75494202c72686a82d4de764618",
    }
    return mapping[label]


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "science_authorized": result["science_authorized"],
                      "loaded_endpoints": result["loaded_endpoints"], "forwards": result["forwards"]}, indent=2))
