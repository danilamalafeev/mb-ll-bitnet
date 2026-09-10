"""Training, atomic checkpoints, held-out evaluation and synchronized timing."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F

from .config import Config
from .data import SampleStream, collate, make_eval_sets, solve
from .model import ReasoningModel
from .runtime import (environment, peak_memory, reset_peak, restore_rng, rng_state,
                      seed_everything, synchronize)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def copy_checkpoint(source, destination):
    """Copy a checkpoint atomically so interruption cannot leave a partial best."""
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def suite_manifest(suites):
    serialized = {name: [asdict(example) for example in examples] for name, examples in suites.items()}
    digest = hashlib.sha256(json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"fingerprint": digest, "sets": serialized}


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    return state


def save_checkpoint(path, model, optimizer, config, stream, progress, fingerprint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1, "config": config.to_dict(), "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "stream": stream.state_dict(),
        "rng": rng_state(), "progress": dict(progress), "suite_fingerprint": fingerprint,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(path, device):
    state = load_checkpoint(path)
    config = Config.from_dict(state["config"])
    model = ReasoningModel(config.model).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, config, state


def make_optimizer(model, config):
    return torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate,
                             weight_decay=config.train.weight_decay, foreach=False)


def trim_log_to_checkpoint(path, update):
    """Discard uncheckpointed rows after a crash, including a partial last row."""
    path = Path(path)
    if not path.exists():
        return
    lines = path.read_text().splitlines()
    retained = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise ValueError(f"corrupt metrics log at line {index + 1}")
        if row["updates"] <= update:
            retained.append(line)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(line + "\n" for line in retained))
    temporary.replace(path)


@torch.inference_mode()
def prediction_controls(examples, model_predictions):
    """Deterministic controls and agreements on exactly the evaluated examples."""
    if len(examples) != len(model_predictions):
        raise ValueError("predictions must align with examples")
    names = ("return_start", "one_hop", "destination_frequency")
    predictions = {name: [] for name in names}
    targets = [example.target for example in examples]
    for example in examples:
        predictions["return_start"].append(example.start)
        predictions["one_hop"].append(solve(example.transitions, example.start, 1))
        counts = {obj: example.transitions.count(obj) for obj in range(len(example.transitions))}
        predictions["destination_frequency"].append(max(counts, key=lambda obj: (counts[obj], -obj)))
    controls = {
        "uniform_guess": {"expected_accuracy": 1 / len(examples[0].transitions)},
    }
    all_predictions = {"model": list(model_predictions), **predictions}
    for name in names:
        guesses = predictions[name]
        controls[name] = {
            "accuracy": sum(a == b for a, b in zip(guesses, targets)) / len(targets),
            "agreement_with_model": sum(a == b for a, b in zip(guesses, model_predictions)) / len(targets),
        }
    controls["destination_frequency"]["tie_break"] = "smallest_object_id"
    controls["destination_frequency"]["all_destinations_equal_frequency"] = all(
        len(set(example.transitions.count(obj) for obj in range(len(example.transitions)))) == 1
        for example in examples
    )
    agreements = []
    keys = list(all_predictions)
    accuracies = {name: sum(a == b for a, b in zip(values, targets)) / len(targets)
                  for name, values in all_predictions.items()}
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            agreements.append({
                "left": left, "right": right,
                "agreement": sum(a == b for a, b in zip(all_predictions[left], all_predictions[right])) / len(targets),
                "equal_aggregate_accuracy": accuracies[left] == accuracies[right],
            })
    controls["pairwise_agreement"] = agreements
    return controls


@torch.inference_mode()
def evaluate_set(model, examples, *, steps, batch_size, device):
    """Timing covers only forward after batch transfer, with one untimed warmup.

    This is amortized batch latency per example, not single-request latency.
    Data generation, host transfer, loss and metric accumulation are excluded.
    """
    if not examples or batch_size <= 0:
        raise ValueError("evaluation requires examples and a positive batch_size")
    was_training = model.training
    model.eval()
    first = collate(examples[:batch_size], device=device)
    model(first["input_ids"], steps=steps)
    synchronize(device)
    reset_peak(device)
    total_loss = correct = count = 0
    inference_seconds = 0.0
    per_hop = {}
    predictions = []
    model_predictions = []
    for offset in range(0, len(examples), batch_size):
        batch = collate(examples[offset:offset + batch_size], device=device)
        synchronize(device)
        start = time.perf_counter()
        logits = model(batch["input_ids"], steps=steps)
        synchronize(device)
        inference_seconds += time.perf_counter() - start
        losses = F.cross_entropy(logits, batch["target"], reduction="none").cpu().tolist()
        guesses = logits.argmax(dim=-1).cpu().tolist()
        model_predictions.extend(guesses)
        targets = batch["target"].cpu().tolist()
        hops = batch["hops"].cpu().tolist()
        for loss, guess, target, hop in zip(losses, guesses, targets, hops):
            bucket = per_hop.setdefault(hop, {"n": 0, "correct": 0, "loss_sum": 0.0})
            bucket["n"] += 1
            bucket["correct"] += int(guess == target)
            bucket["loss_sum"] += loss
            count += 1
            total_loss += loss
            correct += int(guess == target)
            if len(predictions) < 5:
                predictions.append({"hops": hop, "prediction": f"OBJ_{guess}", "target": f"OBJ_{target}"})
    # Collect norms in a separate, untimed pass. Diagnostics use scalar host
    # reads and must not contaminate the synchronized inference measurement.
    norm_sums = None
    norm_examples = 0
    for offset in range(0, len(examples), batch_size):
        diagnostic_batch = collate(examples[offset:offset + batch_size], device=device)
        _, batch_norms = model.forward_diagnostics(diagnostic_batch["input_ids"], steps=steps)
        batch_n = len(diagnostic_batch["target"])
        if norm_sums is None:
            norm_sums = [{key: (value if key == "step" else value * batch_n)
                          for key, value in row.items()} for row in batch_norms]
        else:
            for total, row in zip(norm_sums, batch_norms):
                for key, value in row.items():
                    if key != "step":
                        total[key] += value * batch_n
        norm_examples += batch_n
    result = {
        "steps": steps, "n": count, "loss": total_loss / count, "accuracy": correct / count,
        "inference_seconds": inference_seconds, "ms_per_example": inference_seconds * 1000 / count,
        "batch_size": batch_size, **peak_memory(device),
        "by_hops": [{"hops": hop, "n": data["n"], "accuracy": data["correct"] / data["n"],
                     "loss": data["loss_sum"] / data["n"]} for hop, data in sorted(per_hop.items())],
        "sample_predictions": predictions,
        "controls": prediction_controls(examples, model_predictions),
        "step_norms": [{key: (value if key == "step" else value / norm_examples)
                        for key, value in row.items()} for row in norm_sums],
        "step_norms_timing": "separate untimed evaluation pass over the same examples",
    }
    model.train(was_training)
    return result


def train(config, out, device, resume=None):
    out = Path(out)
    if out.exists() and any(out.iterdir()) and resume is None:
        raise FileExistsError(f"{out} is not empty; use a new directory or --resume")
    state = load_checkpoint(resume) if resume else None
    if state:
        saved_config = Config.from_dict(state["config"])
        # Only total update target may change. Stream, optimizer and architecture
        # must retain their meaning when restoring a run.
        old, new = saved_config.to_dict(), config.to_dict()
        old["train"]["updates"] = new["train"]["updates"]
        if old != new:
            raise ValueError("resume permits changing only train.updates and execution device")
        if config.train.updates <= state["progress"]["updates"]:
            raise ValueError("--updates must exceed completed checkpoint updates")
        if (out / "checkpoint.pt").exists():
            latest = load_checkpoint(out / "checkpoint.pt")
            if latest["progress"]["updates"] != state["progress"]["updates"] or latest["config"] != state["config"]:
                raise ValueError("output contains a different checkpoint; resume into a new directory")
        elif out.exists() and any(out.iterdir()):
            raise FileExistsError("resume output must be empty or contain the restored checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(config.train.seed, config.train.deterministic, config.train.cpu_threads)
    model = ReasoningModel(config.model).to(device)
    optimizer = make_optimizer(model, config)
    stream = SampleStream(config.data, split="train", seed=config.train.seed + 100_000,
                          min_hops=config.data.train_min_hops, max_hops=config.data.train_max_hops)
    suites = make_eval_sets(config.data)
    manifest = suite_manifest(suites)
    if state and manifest["fingerprint"] != state["suite_fingerprint"]:
        raise ValueError("evaluation data changed since checkpoint creation")
    progress = {"updates": 0, "examples_seen": 0, "example_steps": 0,
                "train_seconds": 0.0, "wall_seconds": 0.0,
                "peak_cuda_allocated_mb": None, "peak_cuda_reserved_mb": None,
                "process_peak_rss_mb": 0.0, "best_validation_accuracy": -1.0,
                "best_validation_loss": None, "best_update": 0}
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        stream.load_state_dict(state["stream"])
        progress.update(state["progress"])
        restore_rng(state["rng"])
        trim_log_to_checkpoint(out / "metrics.jsonl", progress["updates"])
        if not (out / "checkpoint_best.pt").exists():
            source_best = Path(resume).with_name("checkpoint_best.pt")
            if source_best.is_file():
                best_state = load_checkpoint(source_best)
                if (Config.from_dict(best_state["config"]) != Config.from_dict(state["config"]) or
                        best_state["suite_fingerprint"] != state["suite_fingerprint"]):
                    raise ValueError("companion checkpoint_best.pt is incompatible with resume checkpoint")
                copy_checkpoint(source_best, out / "checkpoint_best.pt")
            elif progress["best_update"] in (0, progress["updates"]):
                # Legacy checkpoints had no separate best; if the resumed update
                # is recorded as best, the resume payload itself is sufficient.
                copy_checkpoint(resume, out / "checkpoint_best.pt")
            else:
                raise FileNotFoundError(
                    "resume checkpoint records an earlier validation best, but companion "
                    "checkpoint_best.pt is missing"
                )
    write_json(out / "config.json", config.to_dict())
    write_json(out / "environment.json", environment(device))
    write_json(out / "eval_sets.json", manifest)
    session_start = time.perf_counter()
    previous_wall = progress["wall_seconds"]
    model.train()
    loss_sum = accuracy_sum = segment_n = 0
    train_by_hop = {}
    with (out / "metrics.jsonl").open("a") as logfile:
        for update in range(progress["updates"] + 1, config.train.updates + 1):
            reset_peak(device)
            synchronize(device)
            tick = time.perf_counter()
            batch = collate([stream.sample() for _ in range(config.train.batch_size)], device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"])
            per_example_loss = F.cross_entropy(logits, batch["target"], reduction="none")
            loss = per_example_loss.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at update {update}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            synchronize(device)
            progress["train_seconds"] += time.perf_counter() - tick
            for key, value in peak_memory(device).items():
                if value is not None:
                    progress[key] = max(value, progress[key] or 0)
            progress["updates"] = update
            progress["examples_seen"] += config.train.batch_size
            progress["example_steps"] += config.train.batch_size * config.model.steps
            progress["wall_seconds"] = previous_wall + time.perf_counter() - session_start
            loss_sum += loss.item() * config.train.batch_size
            accuracy_sum += (logits.argmax(-1) == batch["target"]).sum().item()
            segment_n += config.train.batch_size
            for hop, item_loss, guess, target in zip(batch["hops"].tolist(), per_example_loss.detach().tolist(),
                                                      logits.argmax(-1).tolist(), batch["target"].tolist()):
                bucket = train_by_hop.setdefault(hop, {"n": 0, "correct": 0, "loss_sum": 0.0})
                bucket["n"] += 1
                bucket["correct"] += int(guess == target)
                bucket["loss_sum"] += item_loss
            if update % config.train.log_every == 0 or update == config.train.updates:
                record = {"event": "train", **progress, "loss": loss_sum / segment_n,
                          "accuracy": accuracy_sum / segment_n, "grad_norm": norm.item(),
                          "by_hops": [{"hops": hop, "n": data["n"],
                                       "accuracy": data["correct"] / data["n"],
                                       "loss": data["loss_sum"] / data["n"]}
                                      for hop, data in sorted(train_by_hop.items())]}
                logfile.write(json.dumps(record, allow_nan=False) + "\n")
                logfile.flush()
                print(f"update {update}/{config.train.updates} loss={record['loss']:.4f} accuracy={record['accuracy']:.3f} train_s={progress['train_seconds']:.2f}", flush=True)
                loss_sum = accuracy_sum = segment_n = 0
                train_by_hop = {}
            if update % config.train.eval_every == 0 or update == config.train.updates:
                metrics = evaluate_set(model, suites["validation"], steps=config.model.steps,
                                       batch_size=config.train.eval_batch_size, device=device)
                logfile.write(json.dumps({"event": "validation", "updates": update, **metrics}, allow_nan=False) + "\n")
                logfile.flush()
                print(f"validation accuracy={metrics['accuracy']:.3f} loss={metrics['loss']:.4f}", flush=True)
                improved = (metrics["accuracy"] > progress["best_validation_accuracy"] or
                            (metrics["accuracy"] == progress["best_validation_accuracy"] and
                             (progress["best_validation_loss"] is None or metrics["loss"] < progress["best_validation_loss"])))
                if improved:
                    progress["best_validation_accuracy"] = metrics["accuracy"]
                    progress["best_validation_loss"] = metrics["loss"]
                    progress["best_update"] = update
                progress["wall_seconds"] = previous_wall + time.perf_counter() - session_start
                save_checkpoint(out / "checkpoint.pt", model, optimizer, config, stream, progress, manifest["fingerprint"])
                if improved:
                    copy_checkpoint(out / "checkpoint.pt", out / "checkpoint_best.pt")
                progress["wall_seconds"] = previous_wall + time.perf_counter() - session_start
    summary = {**progress, "parameters": sum(p.numel() for p in model.parameters()),
               "train_steps": config.model.steps, "quantized": config.model.quantized,
               "seed": config.train.seed, "device": str(device), "suite_fingerprint": manifest["fingerprint"],
               "train_timing": "synchronized data generation + transfer + forward + backward + optimizer; excludes evaluation/checkpoint writes"}
    write_json(out / "train_summary.json", summary)
    return summary


def evaluate_checkpoint(checkpoint, steps, device, out, batch_size=None, splits=None):
    model, config, state = load_model(checkpoint, device)
    seed_everything(config.train.seed, config.train.deterministic, config.train.cpu_threads)
    suites = make_eval_sets(config.data)
    fingerprint = suite_manifest(suites)["fingerprint"]
    if fingerprint != state["suite_fingerprint"]:
        raise ValueError("evaluation suite differs from the checkpoint manifest")
    measurements = []
    for split in (splits or list(suites)):
        for budget in steps:
            result = evaluate_set(model, suites[split], steps=budget,
                                  batch_size=batch_size or config.train.eval_batch_size, device=device)
            measurements.append({"split": split, **result})
            print(f"{split:16s} steps={budget:2d} accuracy={result['accuracy']:.3f} loss={result['loss']:.4f} ms/example={result['ms_per_example']:.3f}", flush=True)
    report = {
        "format_version": 1, "checkpoint": str(Path(checkpoint).resolve()),
        **state["progress"], "seed": config.train.seed, "train_steps": config.model.steps,
        "device": str(device), "quantized": config.model.quantized,
        "num_objects": config.data.num_objects,
        "parameters": sum(p.numel() for p in model.parameters()), "suite_fingerprint": fingerprint,
        "environment": environment(device), "config": config.to_dict(),
        "timing": "forward only, one warmup batch per split/budget, synchronized; ms_per_example is amortized batch latency",
        "measurements": measurements,
    }
    write_json(out, report)
    return report
