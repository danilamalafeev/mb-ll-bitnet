"""Bounded E15 register interpreter implementation.

The module deliberately contains the finite DSL, frozen manifests, two paired
models, and small evaluation/checkpoint helpers in one place.  It is an
implementation/preflight surface: importing it never trains or evaluates a
scientific run.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import AttentionReader, FFNBlock, ModelConfig
from .quantization import BitLinear, make_linear

OPS = ("ADD", "XOR", "SWAP")
OP_TO_ID = {op: i for i, op in enumerate(OPS)}
STATE_ORDER = tuple(itertools.product(range(16), repeat=2))
PRIMARY_UPDATES = 2000
PRIMARY_BATCH = 64
PRIMARY_SCHEDULE = (1, 2, 3) * 666 + (2, 2)
QAT_PARAMETER_COUNT = 152768  # protocol value; see parameter_count() below
GRU_PARAMETER_COUNT = 152720

# This is deliberately a closed set.  A preflight whose evidence cannot prove
# all of these exact sources was frozen is not usable for a scientific arm.
FROZEN_SOURCE_RELATIVE_PATHS = {
    "module": "looped_bitnet/register_e15.py",
    "runner": "scripts/register_interpreter_e15.py",
    "tests": "tests/test_register_e15.py",
    "protocol": "results/E15_REGISTER_IMPLEMENTATION_PROTOCOL.md",
    "semantic_audit": "results/E15_REGISTER_DSL_SEMANTIC_AUDIT.json",
    "semantic_audit_script": "scripts/register_dsl_e15_audit.py",
    "model": "looped_bitnet/model.py",
    "quantization": "looped_bitnet/quantization.py",
    "runtime": "looped_bitnet/runtime.py",
}


def execute_program(program: Sequence[str], state: tuple[int, int]) -> tuple[int, int]:
    x, y = state
    for op in program:
        if op == "ADD":
            x = (x + y) % 16
        elif op == "XOR":
            x ^= y
        elif op == "SWAP":
            x, y = y, x
        else:
            raise ValueError(f"unknown opcode {op!r}")
    return x, y


def forbidden(program: Sequence[str]) -> bool:
    return any(a == "ADD" and b == "XOR" for a, b in zip(program, program[1:]))


def all_programs(lengths: Sequence[int] = (1, 2, 3)) -> list[tuple[str, ...]]:
    return [p for length in lengths for p in itertools.product(OPS, repeat=length)]


def semantic_signature(program: Sequence[str]) -> tuple[tuple[int, int], ...]:
    return tuple(execute_program(program, state) for state in STATE_ORDER)


def state_split() -> dict[str, list[tuple[int, int]]]:
    ordered = sorted(STATE_ORDER, key=lambda s: hashlib.sha256(f"E15-state-v1:{s[0]}:{s[1]}".encode("ascii")).hexdigest())
    return {"train": ordered[:192], "validation": ordered[192:224], "test": ordered[224:]}


def _hash_program(prefix: str, program: Sequence[str]) -> str:
    return hashlib.sha256((prefix + json.dumps(list(program), separators=(",", ":"))).encode()).hexdigest()


def select_secondary(length: int, group: str, limit: int = 32) -> list[tuple[str, ...]]:
    """Select the frozen length-4/6 secondary set by semantic classes."""
    if group not in {"allowed", "forbidden"}:
        raise ValueError("group must be allowed or forbidden")
    candidates = [p for p in itertools.product(OPS, repeat=length) if forbidden(p) == (group == "forbidden")]
    classes: dict[tuple[tuple[int, int], ...], list[tuple[str, ...]]] = {}
    for p in candidates:
        classes.setdefault(semantic_signature(p), []).append(p)
    prefix = f"E15-secondary-v1:{length}:{group}:"
    ordered_classes = sorted(classes.items(), key=lambda row: hashlib.sha256((prefix + json.dumps(row[0], separators=(",", ":"))).encode()).hexdigest())
    for _, members in ordered_classes:
        members.sort(key=lambda p: _hash_program("E15-secondary-member-v1:", p))
    selected: list[tuple[str, ...]] = []
    cursor = 0
    while len(selected) < min(limit, len(candidates)):
        progressed = False
        for _, members in ordered_classes:
            if cursor < len(members):
                selected.append(members[cursor])
                progressed = True
                if len(selected) == min(limit, len(candidates)):
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def secondary_manifest() -> dict[str, list[list[str]]]:
    return {f"length{length}_{group}": [list(p) for p in select_secondary(length, group)]
            for length in (4, 6) for group in ("allowed", "forbidden")}


@dataclass(frozen=True)
class RegisterExample:
    x: int
    y: int
    program: tuple[str, ...]

    @property
    def targets(self) -> tuple[tuple[int, int], ...]:
        state = (self.x, self.y)
        out = []
        for op in self.program:
            state = execute_program((op,), state)
            out.append(state)
        return tuple(out)


def _canonical_batch(batch: Sequence[RegisterExample]) -> list[dict[str, Any]]:
    return [{"x": e.x, "y": e.y, "program": list(e.program)} for e in batch]


def batch_digest(batches: Sequence[Sequence[RegisterExample]]) -> str:
    h = hashlib.sha256()
    for batch in batches:
        h.update(json.dumps(_canonical_batch(batch), separators=(",", ":")).encode())
        h.update(b"\n")
    return h.hexdigest()


def make_paired_batches(seed: int, updates: int = PRIMARY_UPDATES, batch_size: int = PRIMARY_BATCH) -> list[list[RegisterExample]]:
    if updates != PRIMARY_UPDATES or batch_size != PRIMARY_BATCH:
        raise ValueError("E15 paired schedule is fixed at 2000 updates and batch 64")
    rng = random.Random(seed + 100000)
    by_length = {length: [p for p in all_programs((length,)) if not forbidden(p)] for length in (1, 2, 3)}
    split = state_split()["train"]
    batches = []
    for length in PRIMARY_SCHEDULE:
        batches.append([RegisterExample(*rng.choice(split), tuple(rng.choice(by_length[length]))) for _ in range(batch_size)])
    return batches


def protocol_manifest() -> dict[str, Any]:
    programs = all_programs()
    seen = [p for p in programs if not forbidden(p)]
    held = [p for p in programs if forbidden(p)]
    signatures = {p: semantic_signature(p) for p in programs}
    novel = [p for p in held if signatures[p] not in {signatures[q] for q in seen}]
    order_difference = sum(execute_program(("ADD", "XOR"), s) != execute_program(("XOR", "ADD"), s) for s in STATE_ORDER)
    split = state_split()
    return {
        "format_version": 2,
        "experiment": "register_interpreter_e15",
        "state_split": {k: [list(s) for s in v] for k, v in split.items()},
        "state_marginals": {k: {"x": [sum(x == value for x, _ in states) for value in range(16)],
                                 "y": [sum(y == value for _, y in states) for value in range(16)]}
                            for k, states in split.items()},
        "programs": {"all": [list(p) for p in programs], "seen": [list(p) for p in seen],
                     "forbidden": [list(p) for p in held], "primary": [list(p) for p in novel]},
        "controls": {"identity": [["XOR", "XOR"], ["SWAP", "SWAP"]],
                      "equivalent_primary_excluded": {"program": ["ADD", "XOR", "XOR"], "equivalent_to": ["ADD"]},
                      "order_control": {"left": ["ADD", "XOR"], "right": ["XOR", "ADD"], "different_states": order_difference}},
        "secondary": secondary_manifest(),
        "schedule": {"updates": PRIMARY_UPDATES, "batch_size": PRIMARY_BATCH,
                      "lengths": [1, 2, 3], "counts": {str(n): PRIMARY_SCHEDULE.count(n) for n in (1, 2, 3)}},
        "objective": "mean_over_instructions(CE(x)+CE(y))",
        "selection": {"validation_every_updates": 250,
                      "primary": "macro_final_joint_by_length",
                      "tie_break": "lower_macro_final_dual_ce_by_length_then_earlier_update"},
        "optimizer": {"type": "AdamW", "lr": 0.001, "weight_decay": 0.01, "foreach": False, "grad_clip": 1.0},
        "gate": {"primitive": "32/32", "seen_composition": "31/32", "primary": "244/256"},
        "batch_digest_seed0": batch_digest(make_paired_batches(0)),
        "required_source_names": sorted(FROZEN_SOURCE_RELATIVE_PATHS),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_state_dict(model: nn.Module) -> str:
    """Stable digest of names, dtypes, shapes, and CPU tensor bytes."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(str(cpu.dtype).encode("ascii")); digest.update(b"\0")
        digest.update(json.dumps(list(cpu.shape)).encode("ascii")); digest.update(b"\0")
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def validate_manifest_schema(manifest: dict[str, Any]) -> None:
    required = {"format_version", "experiment", "state_split", "programs", "secondary", "schedule",
                "objective", "optimizer", "gate", "source_hashes", "required_source_names"}
    missing = required - set(manifest)
    if missing or manifest.get("format_version") != 2:
        raise ValueError(f"invalid E15 manifest schema; missing={sorted(missing)}")
    expected_sources = set(FROZEN_SOURCE_RELATIVE_PATHS)
    if set(manifest["required_source_names"]) != expected_sources or set(manifest["source_hashes"]) != expected_sources:
        raise ValueError("manifest does not contain the mandatory frozen source set")
    split = manifest["state_split"]
    if {name: len(split.get(name, [])) for name in ("train", "validation", "test")} != {"train": 192, "validation": 32, "test": 32}:
        raise ValueError("manifest state split is invalid")
    programs = manifest["programs"]
    if len(programs.get("seen", [])) != 32 or len(programs.get("primary", [])) != 6:
        raise ValueError("manifest program sets are invalid")
    if manifest["schedule"] != {"updates": 2000, "batch_size": 64, "lengths": [1, 2, 3], "counts": {"1": 666, "2": 668, "3": 666}}:
        raise ValueError("manifest schedule is invalid")


def write_preflight(root: Path) -> Path:
    """Write frozen manifests once; refuse to overwrite a non-empty directory."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty preflight directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = protocol_manifest()
    project = Path(__file__).parents[1]
    manifest["source_hashes"] = {name: sha256_file(project / relative)
                                 for name, relative in FROZEN_SOURCE_RELATIVE_PATHS.items()}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    prefix = [_canonical_batch(batch) for batch in make_paired_batches(0)[:2]]
    (root / "batch_prefix_seed0.json").write_text(json.dumps(prefix, indent=2) + "\n")
    return root / "manifest.json"


class _RegisterBase(nn.Module):
    hidden_size: int

    def _validate(self, x: Tensor, y: Tensor, op_ids: Tensor | None = None) -> None:
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.dtype != torch.long or y.dtype != torch.long:
            raise ValueError("x and y must be matching 1-D torch.long tensors")
        if torch.any((x < 0) | (x > 15)) or torch.any((y < 0) | (y > 15)):
            raise ValueError("register values must be in 0..15")
        if op_ids is not None and (op_ids.ndim != 2 or op_ids.shape[0] != x.shape[0] or op_ids.dtype != torch.long):
            raise ValueError("op_ids must have shape [batch, length] and torch.long dtype")


class QATRegisterModel(_RegisterBase):
    """QAT fixed reader with a persistent hidden register state."""
    hidden_size = 64

    def __init__(self):
        super().__init__()
        cfg = ModelConfig(d_model=64, d_ff=256, num_blocks=4, num_heads=4, num_objects=16,
                          max_tokens=51, steps=4, quantized=True)
        self.x_embedding = nn.Embedding(16, 64)
        self.y_embedding = nn.Embedding(16, 64)
        self.opcode_embedding = nn.Embedding(3, 64)
        self.role_keys = nn.Parameter(torch.empty(2, 64))
        self.memory_norm = nn.LayerNorm(64)
        self.reader = AttentionReader(cfg)
        self.blocks = nn.ModuleList(FFNBlock(cfg) for _ in range(4))
        self.output_norm = nn.LayerNorm(64)
        self.x_head = BitLinear(64, 16)
        self.y_head = BitLinear(64, 16)
        self._initialize()

    def _initialize(self) -> None:
        for emb in (self.x_embedding, self.y_embedding, self.opcode_embedding):
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.role_keys, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_cache(self, x: Tensor, y: Tensor) -> dict[str, Any]:
        self._validate(x, y)
        xv, yv = self.x_embedding(x), self.y_embedding(y)
        key_memory = self.memory_norm(self.role_keys).unsqueeze(0).expand(x.shape[0], -1, -1)
        value_memory = self.memory_norm(torch.stack((xv, yv), dim=1))
        with torch.autocast(device_type=x.device.type, enabled=False):
            kv = self.reader.build_kv(key_memory, value_memory)
        return {"h": xv + yv, "x_initial": xv, "y_initial": yv, "kv": kv, "substeps": 0}

    def step(self, cache: dict[str, Any], opcode: int | Tensor) -> tuple[tuple[Tensor, Tensor], dict[str, Any]]:
        if isinstance(opcode, int):
            opcode = torch.full((cache["h"].shape[0],), opcode, dtype=torch.long, device=cache["h"].device)
        if opcode.ndim != 1 or opcode.shape[0] != cache["h"].shape[0] or torch.any((opcode < 0) | (opcode >= 3)):
            raise ValueError("opcode must be a batch-sized value in 0..2")
        h = cache["h"]
        for inner in range(4):
            h = h + self.opcode_embedding(opcode) / (64 ** 0.5)
            h = h + self.reader(h, cache["kv"])
            h = self.blocks[(cache["substeps"] + inner) % 4](h)
        cache = dict(cache, h=h, substeps=cache["substeps"] + 4)
        logits = self.output(self.output_norm(h))
        return logits, cache

    def output(self, h: Tensor) -> tuple[Tensor, Tensor]:
        return self.x_head(h), self.y_head(h)

    def forward(self, x: Tensor, y: Tensor, op_ids: Tensor) -> tuple[Tensor, Tensor]:
        self._validate(x, y, op_ids)
        cache = self.init_cache(x, y)
        xs, ys = [], []
        for index in range(op_ids.shape[1]):
            (xl, yl), cache = self.step(cache, op_ids[:, index])
            xs.append(xl); ys.append(yl)
        return torch.stack(xs, 1), torch.stack(ys, 1)


class GRURegisterModel(_RegisterBase):
    """Float paired control with the same initial context and opcode stream."""
    hidden_size = 132

    def __init__(self):
        super().__init__()
        self.x_embedding = nn.Embedding(16, 64)
        self.y_embedding = nn.Embedding(16, 64)
        self.opcode_embedding = nn.Embedding(3, 64)
        self.initial_projection = nn.Linear(128, 132, bias=False)
        self.cell = nn.GRUCell(192, 132)
        self.output_norm = nn.LayerNorm(132)
        self.x_head = nn.Linear(132, 16, bias=False)
        self.y_head = nn.Linear(132, 16, bias=False)
        for emb in (self.x_embedding, self.y_embedding, self.opcode_embedding):
            nn.init.normal_(emb.weight, std=0.02)
        for module in (self.initial_projection, self.x_head, self.y_head):
            nn.init.normal_(module.weight, std=0.02)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_cache(self, x: Tensor, y: Tensor) -> dict[str, Any]:
        self._validate(x, y)
        xv, yv = self.x_embedding(x), self.y_embedding(y)
        return {"h": self.initial_projection(torch.cat((xv, yv), dim=-1)), "x_initial": xv, "y_initial": yv, "substeps": 0}

    def step(self, cache: dict[str, Any], opcode: int | Tensor) -> tuple[tuple[Tensor, Tensor], dict[str, Any]]:
        if isinstance(opcode, int):
            opcode = torch.full((cache["h"].shape[0],), opcode, dtype=torch.long, device=cache["h"].device)
        opv = self.opcode_embedding(opcode)
        inputs = torch.cat((cache["x_initial"], cache["y_initial"], opv), dim=-1)
        h = cache["h"]
        for _ in range(4):
            h = self.cell(inputs, h)
        cache = dict(cache, h=h, substeps=cache["substeps"] + 4)
        z = self.output_norm(h)
        return (self.x_head(z), self.y_head(z)), cache

    def forward(self, x: Tensor, y: Tensor, op_ids: Tensor) -> tuple[Tensor, Tensor]:
        self._validate(x, y, op_ids)
        cache = self.init_cache(x, y)
        xs, ys = [], []
        for index in range(op_ids.shape[1]):
            (xl, yl), cache = self.step(cache, op_ids[:, index])
            xs.append(xl); ys.append(yl)
        return torch.stack(xs, 1), torch.stack(ys, 1)


def parameter_report() -> dict[str, int]:
    qat, gru = QATRegisterModel(), GRURegisterModel()
    return {"qat_actual": qat.parameter_count(), "qat_protocol": QAT_PARAMETER_COUNT,
            "gru_actual": gru.parameter_count(), "gru_protocol": GRU_PARAMETER_COUNT}


def loss_for_batch(model: _RegisterBase, batch: Sequence[RegisterExample]) -> Tensor:
    if not batch:
        raise ValueError("empty batch")
    x = torch.tensor([e.x for e in batch], dtype=torch.long)
    y = torch.tensor([e.y for e in batch], dtype=torch.long)
    length = len(batch[0].program)
    if any(len(e.program) != length for e in batch):
        raise ValueError("batch must contain one homogeneous program length")
    ops = torch.tensor([[OP_TO_ID[o] for o in e.program] for e in batch], dtype=torch.long)
    xl, yl = model(x, y, ops)
    tx = torch.tensor([[t[0] for t in e.targets] for e in batch], dtype=torch.long)
    ty = torch.tensor([[t[1] for t in e.targets] for e in batch], dtype=torch.long)
    return (F.cross_entropy(xl.transpose(1, 2), tx) + F.cross_entropy(yl.transpose(1, 2), ty))


def evaluate_program(model: _RegisterBase, program: Sequence[str], states: Sequence[tuple[int, int]], *, include_predictions: bool = False) -> dict[str, Any]:
    batch = [RegisterExample(x, y, tuple(program)) for x, y in states]
    x = torch.tensor([e.x for e in batch], dtype=torch.long)
    y = torch.tensor([e.y for e in batch], dtype=torch.long)
    ops = torch.tensor([[OP_TO_ID[o] for o in program] for _ in batch], dtype=torch.long)
    with torch.inference_mode():
        xl, yl = model(x, y, ops)
    px, py = xl.argmax(-1), yl.argmax(-1)
    targets = torch.tensor([[t for t in e.targets] for e in batch], dtype=torch.long)
    joint = ((px[:, -1] == targets[:, -1, 0]) & (py[:, -1] == targets[:, -1, 1])).tolist()
    prefix = [int(((px[:, i] == targets[:, i, 0]) & (py[:, i] == targets[:, i, 1])).sum()) for i in range(len(program))]
    final_x = (px[:, -1] == targets[:, -1, 0]).tolist()
    final_y = (py[:, -1] == targets[:, -1, 1]).tolist()
    full_trace = sum(all(bool((px[n, i] == targets[n, i, 0]) and (py[n, i] == targets[n, i, 1]))
                         for i in range(len(program))) for n in range(len(batch)))
    row = {"program": list(program), "states": len(states), "joint_final": int(sum(joint)), "prefix_joint": prefix,
           "full_trace": int(full_trace),
           "final_x_correct": int(sum(final_x)), "final_y_correct": int(sum(final_y))}
    if include_predictions:
        row["predictions"] = [{"state": list(states[n]), "target_trace": [list(t) for t in batch[n].targets],
                               "predicted_trace": [[int(px[n, i]), int(py[n, i])] for i in range(len(program))],
                               "prefix_joint_correct": [bool((px[n, i] == targets[n, i, 0]) and (py[n, i] == targets[n, i, 1])) for i in range(len(program))],
                               "joint_final_correct": bool(joint[n])}
                              for n in range(len(batch))]
    return row


def gate_report(rows: Iterable[dict[str, Any]], primitive: Sequence[str] = OPS) -> dict[str, Any]:
    rows = list(rows)
    primitive_set = {tuple((p,)) for p in primitive}
    seen_set = set(all_programs()) - {p for p in all_programs() if forbidden(p)}
    by_program = {tuple(r["program"]): r for r in rows}
    primitive_rows = [by_program[p] for p in sorted(primitive_set) if p in by_program]
    composition_programs = sorted(seen_set - primitive_set)
    composition_rows = [by_program[p] for p in composition_programs if p in by_program]
    valid_counts = all(isinstance(row.get("states"), int) and row["states"] == 32
                       and isinstance(row.get("joint_final"), int) and 0 <= row["joint_final"] <= 32
                       for row in rows)
    complete = len(by_program) == len(rows) == 32 and set(by_program) == seen_set and valid_counts
    passed = complete and len(primitive_rows) == 3 and all(r["joint_final"] == 32 for r in primitive_rows) and all(r["joint_final"] >= 31 for r in composition_rows)
    return {"passed": passed, "complete": complete,
            "primitive": [{"program": r["program"], "joint": r["joint_final"], "required": 32} for r in primitive_rows],
            "composition": [{"program": r["program"], "joint": r["joint_final"], "required": 31} for r in composition_rows]}


def save_checkpoint(path: Path, model: _RegisterBase, *, seed: int, update: int, manifest_hash: str,
                    objective: str = "mean_over_instructions(CE(x)+CE(y))", optimizer: dict[str, Any] | None = None,
                    batch_prefix_digest: str | None = None, config: dict[str, Any] | None = None,
                    tag: str = "selected", initial_model_digest: str | None = None,
                    full_batch_digest: str | None = None, source_hashes: dict[str, str] | None = None,
                    environment: dict[str, Any] | None = None, validation: dict[str, Any] | None = None) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    if tag not in {"selected", "latest", "toy"}:
        raise ValueError("checkpoint tag must be selected, latest, or toy")
    payload = {"format_version": 2, "model": model.__class__.__name__, "state_dict": model.state_dict(),
               "seed": seed, "update": update, "params": model.parameter_count(), "manifest_hash": manifest_hash,
               "objective": objective, "optimizer": optimizer or {}, "config": config or {},
               "tag": tag, "batch_prefix_digest": batch_prefix_digest,
               "initial_model_digest": initial_model_digest, "full_batch_digest": full_batch_digest,
               "source_hashes": source_hashes or {}, "environment": environment or {},
               "validation": validation, "model_digest": digest_state_dict(model)}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, model: _RegisterBase, *, manifest_hash: str | None = None,
                    expected_update: int | None = None, expected_prefix_digest: str | None = None,
                    expected_seed: int | None = None, expected_tag: str | None = None,
                    expected_config: dict[str, Any] | None = None,
                    expected_source_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 2:
        raise ValueError("checkpoint format version mismatch")
    if payload.get("model") != model.__class__.__name__:
        raise ValueError("checkpoint model class mismatch")
    if payload.get("params") != model.parameter_count():
        raise ValueError("checkpoint parameter count mismatch")
    if manifest_hash is not None and payload.get("manifest_hash") != manifest_hash:
        raise ValueError("checkpoint manifest hash mismatch")
    if expected_update is not None and payload.get("update") != expected_update:
        raise ValueError("checkpoint update mismatch")
    if expected_prefix_digest is not None and payload.get("batch_prefix_digest") != expected_prefix_digest:
        raise ValueError("checkpoint batch prefix digest mismatch")
    if expected_seed is not None and payload.get("seed") != expected_seed:
        raise ValueError("checkpoint seed mismatch")
    if expected_tag is not None and payload.get("tag") != expected_tag:
        raise ValueError("checkpoint tag mismatch")
    if expected_config is not None and payload.get("config") != expected_config:
        raise ValueError("checkpoint config mismatch")
    if expected_source_hashes is not None and payload.get("source_hashes") != expected_source_hashes:
        raise ValueError("checkpoint source hashes mismatch")
    model.load_state_dict(payload["state_dict"])
    if payload.get("model_digest") != digest_state_dict(model):
        raise ValueError("checkpoint state digest mismatch")
    return payload


def checkpoint_digest_prefix(batches: Sequence[Sequence[RegisterExample]], update: int) -> str:
    if update < 0 or update > len(batches):
        raise ValueError("update outside batch prefix")
    return batch_digest(batches[:update])
