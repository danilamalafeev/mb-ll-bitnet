#!/usr/bin/env python3
"""E14: guarded training on a fixed mixture of input lengths.

This module contains the protocol and evaluator.  Training is deliberately
behind ``--protocol-cleared``; importing it or running semantic tests never
trains or writes an experiment run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from looped_bitnet.config import Config
from looped_bitnet.data import Example, SampleStream, collate, make_eval_sets, solve, split_for_table
from looped_bitnet.engine import make_optimizer, suite_manifest, write_json
from looped_bitnet.model import ReasoningModel
from looped_bitnet.runtime import environment, peak_memory, seed_everything
from scripts.state_scale_train_e13 import validate_checkpoint_payload as validate_e13_checkpoint

UPDATES, BATCH_SIZE = 2000, 64
HOPS = (1, 2, 3)
EVAL_HOPS = tuple(range(1, 9))
STEPS_PER_HOP = 4
VALIDATION_COUNT, VALIDATION_SEED = 256, 20260912
TEST_COUNT, TEST_SEED = 512, 20260913
TRAIN_STREAM_SEED_OFFSET = 100000
OBJECTIVE = "length_mixture_h1_h2_h3_aux_uniform_weight2"
BASELINE_OBJECTIVE = "final_ce_plus_step4_ce_centered_l2_cycle_boundary"
DEFAULT_OUT = Path("runs/length_mixture_e14")
BASELINE_ROOT = Path("runs/state_scale_train_e13_corrected")
RESULT_PATH = Path("results/LENGTH_MIXTURE_E14.md")
VALIDATION_THRESHOLD = 0.95
REQUIRED_NOVELTY_SCOPES = (
    "e07_eval_sets", "e07_reconstructed_suites", "e08_pairs", "e09_manifest",
    "e10_manifest", "e11_manifest", "e12_manifest", "e13_invalid_manifest",
    "e13_corrected_manifest",
)


def _strict_table(value, *, source: str, index: int) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source}: record {index} has malformed transitions")
    if any(type(x) is not int for x in value):
        raise ValueError(f"{source}: record {index} transitions are not integer-valued")
    table = tuple(value)
    if any(x < 0 or x >= len(table) for x in table):
        raise ValueError(f"{source}: record {index} has an out-of-range transition")
    return table


def _strict_record(record: dict, *, source: str, index: int) -> tuple[int, ...]:
    if not isinstance(record, dict):
        raise ValueError(f"{source}: record {index} is not an object")
    table = _strict_table(record.get("transitions"), source=source, index=index)
    start = record.get("start")
    order = record.get("order")
    if type(start) is not int or not 0 <= start < len(table):
        raise ValueError(f"{source}: record {index} has invalid start")
    if (not isinstance(order, list) or len(order) != len(table) or
            any(type(x) is not int for x in order) or set(order) != set(range(len(table)))):
        raise ValueError(f"{source}: record {index} has invalid order")
    if "hops" in record and (type(record["hops"]) is not int or record["hops"] < 1):
        raise ValueError(f"{source}: record {index} has invalid hops")
    return table


def _strict_records(records, *, source: str) -> set[tuple[int, ...]]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{source}: records is missing, empty, or not a list")
    return {_strict_record(record, source=source, index=i) for i, record in enumerate(records)}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_digest(model: ReasoningModel) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def mixture_schedule(updates: int = UPDATES) -> list[int]:
    """The registered schedule: 666/668/666 updates for h1/h2/h3."""
    if updates != UPDATES:
        raise ValueError("E14 schedule is fixed at 2000 updates")
    schedule = [HOPS[(i - 1) % 3] for i in range(1, 1999)] + [2, 2]
    counts = {h: schedule.count(h) for h in HOPS}
    if counts != {1: 666, 2: 668, 3: 666}:
        raise AssertionError(f"bad E14 schedule counts: {counts}")
    return schedule


def make_examples(records: list[dict], hops: int) -> list[Example]:
    return [Example(tuple(r["transitions"]), r["start"], hops, tuple(r["order"])) for r in records]


def _manifest(config: Config, *, count: int, seed: int, split: str, name: str) -> dict:
    stream = SampleStream(config.data, split, seed, 1, 1)
    records, seen = [], set()
    while len(records) < count:
        e = stream.sample(hops=1)
        if e.transitions in seen:
            raise AssertionError("stream emitted duplicate table")
        seen.add(e.transitions)
        records.append({"transitions": list(e.transitions), "start": e.start, "order": list(e.order)})
    if any(split_for_table(r["transitions"], config.data.split_seed) != split for r in records):
        raise AssertionError(f"{name} manifest includes a non-{split} table")
    obj = {"format_version": 1, "experiment": "length_mixture_e14", "name": name,
           "stream_seed": seed, "count": count, "hops": list(HOPS),
           "config_data": config.data.__dict__, "records": records,
           "all_tables_unique": len(seen) == count,
           "pairing": "same transitions/start/order reused for h1,h2,h3"}
    obj["protocol"] = {"schedule_counts": {"1": 666, "2": 668, "3": 666},
                       "updates": UPDATES, "batch_size": BATCH_SIZE,
                       "loss": "(2/h)*sum CE(readout_4k,f^k(start))",
                       "validation_threshold": VALIDATION_THRESHOLD,
                       "test_hops": list(EVAL_HOPS)}
    obj["fingerprint"] = hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return obj


def make_validation_manifest(config: Config, count: int = VALIDATION_COUNT, seed: int = VALIDATION_SEED) -> dict:
    return _manifest(config, count=count, seed=seed, split="validation", name="validation")


def make_test_manifest(config: Config, count: int = TEST_COUNT, seed: int = TEST_SEED) -> dict:
    return _manifest(config, count=count, seed=seed, split="test", name="test")


def _records_from(path: Path) -> set[tuple[int, ...]]:
    if not path.exists():
        raise ValueError(f"required novelty source is missing: {path}")
    try:
        obj = json.loads(path.read_text())
        return _strict_records(obj["records"], source=str(path))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed required novelty source: {path}: {exc}") from exc


def novelty_scopes() -> dict[str, set[tuple[int, ...]]]:
    scopes = {name: set() for name in REQUIRED_NOVELTY_SCOPES}
    eval_paths = sorted(Path("runs").glob("**/eval_sets.json"))
    if not eval_paths:
        raise ValueError("required E07 eval_sets source is missing")
    for path in eval_paths:
        try:
            obj = json.loads(path.read_text())
            suites = obj["sets"]
            if not isinstance(suites, dict) or not suites:
                raise ValueError("sets is missing or not an object")
            for name, records in suites.items():
                scopes["e07_eval_sets"].update(_strict_records(records, source=f"{path}:sets.{name}"))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed E07 eval_sets source: {path}: {exc}") from exc

    checkpoint = Path("runs/intermediate_supervision_pilot/seed0/aux/checkpoint_best.pt")
    if not checkpoint.exists():
        raise ValueError(f"required E07 reconstruction source is missing: {checkpoint}")
    try:
        cfg_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        cfg = Config.from_dict(cfg_state["config"])
        for name, suite in make_eval_sets(cfg.data).items():
            scopes["e07_reconstructed_suites"].update(
                _strict_record({"transitions": list(e.transitions), "start": e.start, "order": list(e.order)},
                               source=f"reconstructed:{name}", index=i)
                for i, e in enumerate(suite))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise ValueError(f"malformed E07 reconstructed suites: {exc}") from exc

    pair = Path("results/LATE_MEMORY_CAUSAL_CHECK.pairs.json")
    if not pair.exists():
        raise ValueError(f"required E08 pairs source is missing: {pair}")
    try:
        records = json.loads(pair.read_text())["records"]
        if not isinstance(records, list) or not records:
            raise ValueError("records is missing or empty")
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"pair record {i} is not an object")
            for side in ("a", "b"):
                scopes["e08_pairs"].add(_strict_record(record[side], source=f"{pair}:{side}", index=i))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed E08 pairs source: {pair}: {exc}") from exc

    source_paths = {
        "e09_manifest": Path("runs/depth_transfer_e09/manifest.json"),
        "e10_manifest": Path("runs/transition_trace_e10/source_manifest.json"),
        "e11_manifest": Path("runs/state_scale_e11/manifest.json"),
        "e12_manifest": Path("runs/state_scale_schedule_e12/manifest.json"),
        "e13_invalid_manifest": Path("runs/state_scale_train_e13/manifest.json"),
        "e13_corrected_manifest": BASELINE_ROOT / "manifest.json",
    }
    for name, path in source_paths.items():
        if not path.exists():
            raise ValueError(f"required {name} source is missing: {path}")
        try:
            obj = json.loads(path.read_text())
            if name == "e10_manifest":
                referenced = obj["path"]
                if not isinstance(referenced, str):
                    raise ValueError("referenced manifest path is not a string")
                referenced_path = Path(referenced)
                if not referenced_path.exists():
                    raise ValueError(f"referenced manifest missing: {referenced_path}")
                obj = json.loads(referenced_path.read_text())
            scopes[name].update(_strict_records(obj["records"], source=str(path)))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed required {name} source: {path}: {exc}") from exc

    missing = [name for name in REQUIRED_NOVELTY_SCOPES if not scopes[name]]
    if missing:
        raise ValueError(f"required novelty sources missing or empty: {missing}")
    return scopes


def novelty_source_paths() -> list[Path]:
    """Return every persisted file consulted by the mandatory novelty audit."""
    paths = set(sorted(Path("runs").glob("**/eval_sets.json")))
    paths.update({Path("runs/intermediate_supervision_pilot/seed0/aux/checkpoint_best.pt"),
                  Path("results/LATE_MEMORY_CAUSAL_CHECK.pairs.json")})
    pointers = {Path("runs/transition_trace_e10/source_manifest.json")}
    paths.update(pointers)
    for pointer in pointers:
        if pointer.exists():
            try:
                referenced = json.loads(pointer.read_text()).get("path")
                if isinstance(referenced, str):
                    paths.add(Path(referenced))
            except (OSError, TypeError, json.JSONDecodeError):
                pass
    paths.update({Path("runs/depth_transfer_e09/manifest.json"), Path("runs/state_scale_e11/manifest.json"),
                  Path("runs/state_scale_schedule_e12/manifest.json"), Path("runs/state_scale_train_e13/manifest.json"),
                  BASELINE_ROOT / "manifest.json"})
    return sorted(paths)


def _digest_examples(examples: list[Example]) -> str:
    h = hashlib.sha256()
    for off in range(0, len(examples), BATCH_SIZE):
        b = collate(examples[off:off + BATCH_SIZE])
        h.update(b["input_ids"].contiguous().numpy().tobytes())
        h.update(b["target"].contiguous().numpy().tobytes())
    return h.hexdigest()


def stream_digest(config: Config, *, updates: int = UPDATES, seed: int) -> str:
    stream = SampleStream(config.data, "train", seed + TRAIN_STREAM_SEED_OFFSET, 2, 2)
    return _digest_examples([stream.sample(hops=2) for _ in range(updates * BATCH_SIZE)])


def mixture_digests(config: Config, seed: int) -> dict:
    schedule = mixture_schedule()
    stream = SampleStream(config.data, "train", seed + TRAIN_STREAM_SEED_OFFSET, 2, 2)
    all_examples, by_h = [], {h: [] for h in HOPS}
    for h in schedule:
        batch = [stream.sample(hops=2) for _ in range(BATCH_SIZE)]
        converted = [Example(e.transitions, e.start, h, e.order) for e in batch]
        all_examples.extend(converted); by_h[h].extend(converted)
    return {"base_h2_stream_digest": stream_digest(config, seed=seed),
            "mixture_input_target_digest": _digest_examples(all_examples),
            "per_hop_digests": {str(h): _digest_examples(v) for h, v in by_h.items()},
            "counts": {str(h): len(v) // BATCH_SIZE for h, v in by_h.items()},
            "sum_h": sum(h for h in schedule for _ in range(BATCH_SIZE)),
            "schedule_digest": hashlib.sha256(bytes(schedule)).hexdigest()}


def validate_config(config: Config) -> None:
    expected = {"d_model": 64, "d_ff": 256, "num_blocks": 4, "num_heads": 4,
                "steps": 8, "quantized": True, "structured_reader": True}
    for key, value in expected.items():
        if getattr(config.model, key) != value:
            raise ValueError(f"E14 requires model.{key}={value}")
    if config.data.train_min_hops != 2 or config.data.train_max_hops != 2:
        raise ValueError("E14 requires the original fixed two-hop SampleStream")
    if config.train.updates != UPDATES or config.train.batch_size != BATCH_SIZE or config.train.eval_every != 250:
        raise ValueError("E14 requires updates=2000, batch_size=64, eval_every=250")
    if config.model.max_tokens < 3 * config.data.num_objects + 3 + 8:
        raise ValueError("model.max_tokens cannot encode test input h8")


def mixture_loss(logits: dict[int, torch.Tensor], examples: list[Example], h: int) -> torch.Tensor:
    if set(logits) != set(range(4, 4 * h + 1, 4)):
        raise ValueError("loss requires every 4k readout through the requested h")
    total = 0.0
    for k in range(1, h + 1):
        target = torch.tensor([solve(e.transitions, e.start, k) for e in examples], device=next(iter(logits.values())).device)
        total = total + F.cross_entropy(logits[4 * k], target)
    return (2.0 / h) * total


def optimizer_metadata(config: Config) -> dict:
    return {"type": "AdamW", "learning_rate": config.train.learning_rate,
            "weight_decay": config.train.weight_decay, "foreach": False,
            "grad_clip": config.train.grad_clip}


def save_tagged_checkpoint(path: Path, model: ReasoningModel, config: Config, *, update: int,
                           validation: dict, initial_digest: str, data_digest: str,
                           suite_fingerprint: str, optimizer_config: dict) -> None:
    payload = {"format_version": 1, "diagnostic_checkpoint": True, "resume_supported": False,
               "objective": OBJECTIVE, "dynamics": {"input_hops": list(HOPS),
               "recurrent_budget": "4*h", "normalization": "none", "teacher_forcing": False},
               "config": config.to_dict(), "model": model.state_dict(),
               "progress": {"updates": update, "validation": validation},
               "selection": "macro final validation accuracy over h1,h2,h3, then mean final CE",
               "seed": config.train.seed, "initial_state_digest": initial_digest,
               "data_stream_digest": data_digest, "input_target_sequence_digest": data_digest,
               "suite_fingerprint": suite_fingerprint, "optimizer": optimizer_config,
               "parameters": sum(value.numel() for value in model.state_dict().values())}
    tmp = path.with_suffix(path.suffix + ".tmp"); torch.save(payload, tmp); tmp.replace(path)


def validate_checkpoint_payload(payload: dict, *, path: Path, seed: int, config: Config) -> None:
    if payload.get("format_version") != 1 or payload.get("diagnostic_checkpoint") is not True:
        raise ValueError(f"checkpoint is not a tagged diagnostic checkpoint: {path}")
    if payload.get("resume_supported") is not False or payload.get("objective") != OBJECTIVE:
        raise ValueError(f"checkpoint objective/resume guard mismatch: {path}")
    if payload.get("seed") != seed or payload.get("config") != {**config.to_dict(), "train": {**config.to_dict()["train"], "seed": seed}}:
        raise ValueError(f"checkpoint config/seed mismatch: {path}")
    dynamics = payload.get("dynamics", {})
    if (dynamics.get("input_hops") != list(HOPS) or dynamics.get("recurrent_budget") != "4*h" or
            dynamics.get("normalization") != "none" or dynamics.get("teacher_forcing") is not False):
        raise ValueError(f"checkpoint dynamics mismatch: {path}")


def validate_baseline_checkpoint(path: Path, seed: int, config: Config) -> tuple[ReasoningModel, dict]:
    if not path.exists(): raise ValueError(f"missing E13 corrected baseline: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_e13_checkpoint(payload, path=path, trained_arm="baseline", seed=seed, config=config)
    if payload.get("seed") != seed or payload.get("config") != {**config.to_dict(), "train": {**config.to_dict()["train"], "seed": seed}}:
        raise ValueError(f"incompatible E13 baseline config: {path}")
    progress = payload.get("progress", {})
    if progress.get("updates") != UPDATES: raise ValueError("E13 baseline is not a 2000-update checkpoint")
    model = ReasoningModel(config.model); model.load_state_dict(payload["model"]); model.eval()
    return model, payload


def baseline_controls(config: Config) -> dict:
    """Validate every reused E13 baseline before any E14 training/evaluation."""
    controls = {}
    for seed in (0, 1, 2):
        path = BASELINE_ROOT / f"seed{seed}" / "baseline" / "checkpoint_best.pt"
        model, payload = validate_baseline_checkpoint(path, seed, config)
        report_path = path.parent / "report.json"
        if not report_path.exists():
            raise ValueError(f"missing E13 baseline report: {report_path}")
        report = json.loads(report_path.read_text())
        if report.get("checkpoint_best_sha256") != sha256_file(path):
            raise ValueError(f"E13 baseline checkpoint hash mismatch: {path}")
        if report.get("updates") != UPDATES or report.get("best_validation_update") is None:
            raise ValueError(f"E13 baseline report update/selection mismatch: {report_path}")
        if payload.get("progress", {}).get("updates") != report.get("best_validation_update"):
            raise ValueError(f"E13 selected update mismatch: {report_path}")
        if report.get("optimizer") != payload.get("optimizer"):
            raise ValueError(f"E13 baseline optimizer metadata mismatch: {report_path}")
        seed_everything(seed, config.train.deterministic, config.train.cpu_threads)
        expected_initial = state_digest(ReasoningModel(config.model))
        expected_stream = stream_digest(config, seed=seed)
        if payload.get("initial_state_digest") != expected_initial or report.get("initial_state_digest") != expected_initial:
            raise ValueError(f"E13 baseline initial digest mismatch: seed{seed}")
        if payload.get("data_stream_digest") != expected_stream or report.get("data_stream_digest") != expected_stream:
            raise ValueError(f"E13 baseline h2 stream digest mismatch: seed{seed}")
        controls[str(seed)] = {"path": str(path.resolve()), "sha256": sha256_file(path),
                               "objective": payload["objective"], "updates": payload["progress"]["updates"],
                               "selected_update": report["best_validation_update"],
                               "initial_state_digest": expected_initial, "data_stream_digest": expected_stream,
                               "parameters": sum(p.numel() for p in model.parameters())}
    return controls


def _native_eval(model, examples_by_h: dict[int, list[Example]], config: Config) -> dict:
    out = {}
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for h, examples in examples_by_h.items():
            preds = {str(s): [] for s in range(4, 4 * h + 1, 4)}; loss = 0.0
            for off in range(0, len(examples), config.train.eval_batch_size):
                chunk = examples[off:off + config.train.eval_batch_size]; b = collate(chunk, device="cpu")
                _, ro = model.forward_with_readouts(b["input_ids"], tuple(int(s) for s in preds), steps=4 * h)
                for s, logits in ro.items(): preds[str(s)].extend(logits.argmax(-1).tolist())
                loss += F.cross_entropy(ro[4 * h], b["target"], reduction="sum").item()
            counts = {str(k): sum(p == solve(e.transitions, e.start, k) for p, e in zip(preds[str(4*k)], examples)) for k in range(1, h + 1)}
            out[str(h)] = {"n": len(examples), "final": {"accuracy": counts[str(h)] / len(examples), "correct": counts[str(h)], "loss": loss / len(examples)},
                           "readouts": {str(k): {"accuracy": counts[str(k)] / len(examples), "correct": counts[str(k)], "n": len(examples)} for k in range(1, h + 1)},
                           "predictions": preds}
    model.train(was_training)
    return out


def validation_score(native: dict) -> tuple[float, float]:
    return (sum(native[str(h)]["final"]["accuracy"] for h in HOPS) / len(HOPS),
            -sum(native[str(h)]["final"]["loss"] for h in HOPS) / len(HOPS))


def validation_gate(native: dict) -> bool:
    return all(native[str(h)]["readouts"][str(k)]["accuracy"] >= VALIDATION_THRESHOLD for h in HOPS for k in range(1, h + 1))


def evaluate_explicit(model: ReasoningModel, records: list[dict], config: Config) -> dict:
    examples_by_h = {h: make_examples(records, h) for h in EVAL_HOPS}
    return _native_eval(model, examples_by_h, config)


def load_mixture_checkpoint(path: Path, seed: int, config: Config, report: dict | None = None) -> tuple[ReasoningModel, dict]:
    if not path.exists(): raise ValueError(f"missing E14 checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint_payload(payload, path=path, seed=seed, config=config)
    selected_update = payload.get("progress", {}).get("updates")
    if report is not None:
        if report.get("checkpoint_best_sha256") != sha256_file(path):
            raise ValueError(f"E14 selected checkpoint hash mismatch: {path}")
        if selected_update != report.get("best_validation_update"):
            raise ValueError(f"E14 selected update mismatch: {path}")
        if not any(row.get("update") == selected_update for row in report.get("validation_history", [])):
            raise ValueError(f"E14 selected update absent from validation history: {path}")
        if payload.get("suite_fingerprint") != report.get("suite_fingerprint"):
            raise ValueError(f"E14 validation suite fingerprint mismatch: {path}")
        for field in ("initial_state_digest", "data_stream_digest", "input_target_sequence_digest",
                      "optimizer", "parameters"):
            if payload.get(field) != report.get(field):
                raise ValueError(f"E14 {field} mismatch between checkpoint and report: {path}")
        if report.get("checkpoint_dynamics") != payload.get("dynamics"):
            raise ValueError(f"E14 checkpoint dynamics mismatch: {path}")
    if not isinstance(selected_update, int) or not 1 <= selected_update <= UPDATES:
        raise ValueError(f"E14 checkpoint has invalid selected update: {path}")
    model = ReasoningModel(config.model); model.load_state_dict(payload["model"]); model.eval()
    return model, payload


def test_predicates(mixture: dict, baseline: dict) -> dict:
    """Compute preregistered h8 delta and strong all-k criterion from counts."""
    primary, strong = {}, True
    for seed in (0, 1, 2):
        m, b = mixture[str(seed)], baseline[str(seed)]
        delta = m["8"]["final"]["correct"] - b["8"]["final"]["correct"]
        primary[str(seed)] = {"mixture": m["8"]["final"]["correct"], "baseline": b["8"]["final"]["correct"],
                              "delta": delta, "threshold": 26 if seed in (0, 1) else 0,
                              "pass": delta >= (26 if seed in (0, 1) else 0)}
        strong = strong and all(m[str(h)]["final"]["correct"] >= 487 for h in range(4, 9))
    return {"primary_h8": primary, "strong_all_h4_h8_all_seeds": strong}


def paired_test_metrics(mixture: dict, baseline: dict, records: list[dict]) -> dict:
    result = {}
    for h in EVAL_HOPS:
        mp, bp = mixture[str(h)]["predictions"][str(4 * h)], baseline[str(h)]["predictions"][str(4 * h)]
        examples = make_examples(records, h)
        result[str(h)] = {"wins": sum(a == solve(e.transitions, e.start, h) and b != solve(e.transitions, e.start, h)
                                         for a, b, e in zip(mp, bp, examples)),
                          "losses": sum(a != solve(e.transitions, e.start, h) and b == solve(e.transitions, e.start, h)
                                           for a, b, e in zip(mp, bp, examples)),
                          "ties": sum((a == solve(e.transitions, e.start, h)) == (b == solve(e.transitions, e.start, h))
                                       for a, b, e in zip(mp, bp, examples)), "count": len(mp)}
    return result


def train_arm(config: Config, out: Path, validation_records: list[dict], device: torch.device,
              validation_fingerprint: str) -> dict:
    validate_config(config); seed_everything(config.train.seed, config.train.deterministic, config.train.cpu_threads)
    model = ReasoningModel(config.model).to(device); initial = state_digest(model)
    stream = SampleStream(config.data, "train", config.train.seed + TRAIN_STREAM_SEED_OFFSET, 2, 2)
    optimizer = make_optimizer(model, config); schedule = mixture_schedule(); history = []; best_score = (float("-inf"), float("-inf")); best_update = None
    expected_baseline = json.loads((BASELINE_ROOT / f"seed{config.train.seed}" / "baseline" / "report.json").read_text())
    if initial != expected_baseline["initial_state_digest"]:
        raise ValueError("E14 initial state does not match paired E13 baseline")
    out.mkdir(parents=True, exist_ok=True); base_h2_h = hashlib.sha256(); mix_h = hashlib.sha256(); schedule_h = hashlib.sha256(); started = time.perf_counter()
    suite_fp = validation_fingerprint
    arm_report_path = out / "report.json"
    for update, h in enumerate(schedule, 1):
        original = [stream.sample(hops=2) for _ in range(BATCH_SIZE)]
        examples = [Example(e.transitions, e.start, h, e.order) for e in original]
        base_b = collate(original); b = collate(examples, device=device)
        for digest, field in ((base_h2_h, "input_ids"), (base_h2_h, "target"), (mix_h, "input_ids"), (mix_h, "target")):
            digest.update((base_b if digest is base_h2_h else b)[field].cpu().contiguous().numpy().tobytes())
        schedule_h.update(bytes([h]))
        optimizer.zero_grad(set_to_none=True); _, ro = model.forward_with_readouts(b["input_ids"], tuple(range(4, 4*h + 1, 4)), steps=4*h)
        loss = mixture_loss(ro, examples, h)
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss at update {update}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip, error_if_nonfinite=True); optimizer.step()
        if update % config.train.eval_every == 0 or update == UPDATES:
            native = _native_eval(model, {h: make_examples(validation_records, h) for h in HOPS}, config)
            score = validation_score(native); history.append({"update": update, "score": score, "native": native})
            if score > best_score:
                best_score, best_update = score, update
                save_tagged_checkpoint(out / "checkpoint_best.pt", model, config, update=update, validation=native,
                                       initial_digest=initial, data_digest=mix_h.hexdigest(), suite_fingerprint=suite_fp,
                                       optimizer_config=optimizer_metadata(config))
            save_tagged_checkpoint(out / "checkpoint.pt", model, config, update=update, validation=native,
                                   initial_digest=initial, data_digest=mix_h.hexdigest(), suite_fingerprint=suite_fp,
                                   optimizer_config=optimizer_metadata(config))
            write_json(arm_report_path, {"status": "running", "seed": config.train.seed, "objective": OBJECTIVE,
                "updates": update, "examples_seen": update * BATCH_SIZE, "steps_seen": update * BATCH_SIZE * 8,
                "best_validation_update": best_update, "validation_history": history,
                "initial_state_digest": initial, "base_h2_stream_digest": base_h2_h.hexdigest(),
                "data_stream_digest": mix_h.hexdigest(), "input_target_sequence_digest": mix_h.hexdigest(),
                "schedule_digest": schedule_h.hexdigest(), "checkpoint_best_sha256": sha256_file(out / "checkpoint_best.pt") if (out / "checkpoint_best.pt").exists() else None,
                "checkpoint_latest_sha256": sha256_file(out / "checkpoint.pt"), "suite_fingerprint": suite_fp,
                "optimizer": optimizer_metadata(config), "parameters": sum(p.numel() for p in model.parameters()),
                "checkpoint_dynamics": {"input_hops": list(HOPS), "recurrent_budget": "4*h", "normalization": "none", "teacher_forcing": False}})
    if best_update is None: raise RuntimeError("no validation checkpoint was selected")
    expected_base = stream_digest(config, seed=config.train.seed)
    independent = mixture_digests(config, config.train.seed)
    if base_h2_h.hexdigest() != expected_base:
        raise AssertionError("actual base-h2 digest differs from E13 stream digest")
    if mix_h.hexdigest() != independent["mixture_input_target_digest"]:
        raise AssertionError("actual mixture digest differs from independent reconstruction")
    payload = torch.load(out / "checkpoint_best.pt", map_location="cpu", weights_only=True)
    return {"status": "complete", "seed": config.train.seed, "objective": OBJECTIVE, "updates": UPDATES,
            "examples_seen": UPDATES * BATCH_SIZE, "sum_h": sum(h for h in schedule) * BATCH_SIZE,
            "initial_state_digest": initial, "base_h2_stream_digest": base_h2_h.hexdigest(), "data_stream_digest": mix_h.hexdigest(),
            "mixture_digests": independent, "schedule_digest": schedule_h.hexdigest(), "best_validation_update": best_update,
            "suite_fingerprint": suite_fp,
            "input_target_sequence_digest": mix_h.hexdigest(),
            "best_validation": payload["progress"]["validation"], "validation_history": history,
            "checkpoint_best_sha256": sha256_file(out / "checkpoint_best.pt"), "checkpoint_latest_sha256": sha256_file(out / "checkpoint.pt"), "checkpoint_objective": payload["objective"],
            "parameters": sum(p.numel() for p in model.parameters()), "optimizer": optimizer_metadata(config),
            "checkpoint_dynamics": payload["dynamics"], "seconds": time.perf_counter() - started}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--protocol-cleared", action="store_true"); p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(); config = Config.load("configs/cycle16_two_hop_steps8_structured_reader.json"); validate_config(config)
    if args.out.exists() and any(args.out.iterdir()): raise FileExistsError("E14 output exists and is non-empty")
    if RESULT_PATH.exists(): raise FileExistsError(f"E14 result already exists: {RESULT_PATH}")
    validation = make_validation_manifest(config); test = make_test_manifest(config); scopes = novelty_scopes()
    novelty_paths = novelty_source_paths()
    if any(not path.exists() for path in novelty_paths):
        raise ValueError(f"novelty provenance source is missing: {[str(p) for p in novelty_paths if not p.exists()]}")
    for name, manifest in (("validation", validation), ("test", test)):
        overlap = {k: len({tuple(r["transitions"]) for r in manifest["records"]} & v) for k, v in scopes.items()}
        if any(overlap.values()): raise ValueError(f"{name} manifest overlaps historical tables: {overlap}")
        manifest["novelty_overlap_counts"] = overlap
        unsigned = {k: v for k, v in manifest.items() if k != "fingerprint"}
        manifest["fingerprint"] = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True); write_json(args.out / "validation_manifest.json", validation); write_json(args.out / "test_manifest.json", test)
    if not args.protocol_cleared:
        print(json.dumps({"validation_manifest": validation["fingerprint"], "test_manifest": test["fingerprint"], "training": "gated"})); return
    baseline_meta = baseline_controls(config)
    protocol = {"format_version": 1, "experiment": "length_mixture_e14", "objective": OBJECTIVE,
                "schedule": {"updates": UPDATES, "counts": {str(h): mixture_schedule().count(h) for h in HOPS},
                             "sum_h": 4000, "mean_recurrent_steps": 8},
                "loss": "(2/h)*sum_{k=1..h} CE(readout_4k, f^k(start))", "batch_size": BATCH_SIZE,
                "validation": {"manifest_fingerprint": validation["fingerprint"], "cadence": 250,
                                "selection": "macro final accuracy h1,h2,h3, then mean final CE, earliest tie",
                                "gate_cells": ["h1@4", "h2@4", "h2@8", "h3@4", "h3@8", "h3@12"],
                                "threshold": VALIDATION_THRESHOLD, "seed0_controls_extension": True},
                "test": {"manifest_fingerprint": test["fingerprint"], "hops": list(EVAL_HOPS), "budgets": [4*h for h in EVAL_HOPS]},
                "predicates": {"primary_min_delta": {"0": 26, "1": 26, "2": 0},
                               "strong_min_correct": 487, "strong_hops": [4, 5, 6, 7, 8]},
                "source_hashes": {"config": sha256_file(Path("configs/cycle16_two_hop_steps8_structured_reader.json")),
                                  "script": sha256_file(Path(__file__)), "tests": sha256_file(Path("tests/test_length_mixture_e14.py")),
                                  "validation_manifest": sha256_file(args.out / "validation_manifest.json"),
                                  "test_manifest": sha256_file(args.out / "test_manifest.json"),
                                  "preflight_snapshot": sha256_file(Path("results/LENGTH_MIXTURE_E14_PREFLIGHT.json")),
                                  "novelty_sources": {str(path): sha256_file(path) for path in novelty_paths},
                                  "baseline_checkpoints": {s: v["sha256"] for s, v in baseline_meta.items()}},
                "baseline_controls": baseline_meta}
    write_json(args.out / "protocol.json", protocol)
    preflight = {"format_version": 1, "experiment": "length_mixture_e14", "status": "before_training",
                 "protocol": protocol, "validation_manifest": validation["fingerprint"],
                 "test_manifest": test["fingerprint"],
                 "novelty_scope_counts": {name: len(values) for name, values in scopes.items()},
                 "source_hashes": protocol["source_hashes"]}
    write_json(args.out / "preflight.json", preflight)
    reports = {}; gate = None
    for seed in (0, 1, 2):
        cfg = Config.from_dict({**config.to_dict(), "train": {**config.to_dict()["train"], "seed": seed}})
        reports[str(seed)] = train_arm(cfg, args.out / f"seed{seed}" / "mixture", validation["records"], torch.device("cpu"), validation["fingerprint"])
        native = reports[str(seed)]["best_validation"]
        if seed == 0:
            gate = validation_gate(native)
            if not gate: break
    mixture_test, baseline_test = {}, {}
    for seed in reports:
        cfg = Config.from_dict({**config.to_dict(), "train": {**config.to_dict()["train"], "seed": int(seed)}})
        mixture_model, _ = load_mixture_checkpoint(args.out / f"seed{seed}" / "mixture" / "checkpoint_best.pt", int(seed), cfg, reports[seed])
        baseline_model, _ = validate_baseline_checkpoint(Path(baseline_meta[seed]["path"]), int(seed), cfg)
        mixture_test[seed] = evaluate_explicit(mixture_model, test["records"], cfg)
        baseline_test[seed] = evaluate_explicit(baseline_model, test["records"], cfg)
        reports[seed]["paired_test_metrics"] = paired_test_metrics(mixture_test[seed], baseline_test[seed], test["records"])
    predicates = test_predicates(mixture_test, baseline_test) if set(reports) == {"0", "1", "2"} else {"unavailable": "seed0 gate failed"}
    report = {"format_version": 1, "experiment": "length_mixture_e14", "validation_manifest": validation,
              "test_manifest": test, "arms": reports, "gate": {"seed0_pass": gate, "decision": "extend" if gate else "stop_at_seed0"},
              "baseline_controls": baseline_meta, "test_evaluations": {"mixture": mixture_test, "baseline": baseline_test},
              "predicates": predicates,
              "objective": OBJECTIVE, "budget": {"updates_per_arm": UPDATES, "batch_size": BATCH_SIZE,
              "schedule_counts": {str(h): mixture_schedule().count(h) for h in HOPS}, "sum_h": 4000},
              "protocol": protocol, "preflight": preflight, "preflight_sha256": sha256_file(args.out / "preflight.json"),
              "environment": environment(torch.device("cpu")), "peak_memory": peak_memory(torch.device("cpu"))}
    write_json(args.out / "report.json", report)
    RESULT_PATH.write_text(f"# E14 length mixture\n\nMachine-readable report: `{args.out / 'report.json'}`.\n")


if __name__ == "__main__": main()
