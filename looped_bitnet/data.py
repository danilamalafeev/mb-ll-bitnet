"""Online pointer chasing with table-disjoint, reproducible held-out sets.

Each table is a total function on ``num_objects``; cycle mode restricts it to one
directed cycle through every object. Dataset ownership hashes only that function,
never the listing order, start, or requested number of hops. A
seeded finite-domain permutation visits each table at most once per stream,
without an ever-growing set of previously generated examples. Streams explicitly
raise on exhaustion; use enough objects for the planned training run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Any, Sequence

import torch

PAD, EDGE, START, STEP, END = range(5)
OBJ_OFFSET = 5


@dataclass(frozen=True)
class DataConfig:
    num_objects: int = 16
    train_min_hops: int = 1
    train_max_hops: int = 4
    test_max_hops: int = 8
    val_size: int = 256
    test_size: int = 512
    split_seed: int = 1729
    max_tokens: int = 256
    table_mode: str = "functional"

    def __post_init__(self) -> None:
        if self.num_objects < 2:
            raise ValueError("num_objects must be at least 2")
        if not 1 <= self.train_min_hops <= self.train_max_hops < self.test_max_hops:
            raise ValueError("require 1 <= train_min_hops <= train_max_hops < test_max_hops")
        if self.val_size < 1 or self.test_size < 1:
            raise ValueError("validation and test sizes must be positive")
        if self.max_sequence_length > self.max_tokens:
            raise ValueError("the longest encoded example exceeds max_tokens")
        if self.table_mode not in {"functional", "cycle"}:
            raise ValueError("table_mode must be functional or cycle")
        if self.table_mode == "cycle" and self.test_max_hops >= self.num_objects:
            raise ValueError("cycle mode requires test_max_hops < num_objects")

    @property
    def vocab_size(self) -> int:
        return OBJ_OFFSET + self.num_objects

    @property
    def max_sequence_length(self) -> int:
        return 3 * self.num_objects + 3 + self.test_max_hops


def solve(transitions: Sequence[int], start: int, hops: int) -> int:
    """Reference solver; zero hops returns the start object."""
    if not transitions or any(not 0 <= dst < len(transitions) for dst in transitions):
        raise ValueError("transitions must be a nonempty total function")
    if not 0 <= start < len(transitions) or hops < 0:
        raise ValueError("invalid start or hop count")
    current = start
    for _ in range(hops):
        current = transitions[current]
    return current


@dataclass(frozen=True)
class Example:
    transitions: tuple[int, ...]
    start: int
    hops: int
    order: tuple[int, ...]

    def __post_init__(self) -> None:
        # Accept JSON-deserialized lists without changing canonical identities.
        object.__setattr__(self, "transitions", tuple(self.transitions))
        object.__setattr__(self, "order", tuple(self.order))
        solve(self.transitions, self.start, self.hops)
        if sorted(self.order) != list(range(len(self.transitions))):
            raise ValueError("order must list each source object exactly once")

    @property
    def target(self) -> int:
        return solve(self.transitions, self.start, self.hops)

    @property
    def canonical_identity(self) -> tuple[tuple[int, ...], int, int]:
        """Example identity deliberately ignores arbitrary edge listing order."""
        return self.transitions, self.start, self.hops


def encode(example: Example) -> list[int]:
    """Encode condition only; neither answer nor trajectory is read or appended.

    The answer's object symbol can legitimately occur in the complete edge table
    or as the start object. Absence of that symbol is not a leakage criterion.
    """
    tokens: list[int] = []
    for source in example.order:
        tokens.extend((EDGE, OBJ_OFFSET + source, OBJ_OFFSET + example.transitions[source]))
    tokens.extend((START, OBJ_OFFSET + example.start))
    tokens.extend([STEP] * example.hops)
    tokens.append(END)
    return tokens


def split_for_table(transitions: Sequence[int], split_seed: int = 1729) -> str:
    """Assign a canonical table to train/validation/test with 80/10/10 weights."""
    canonical = json.dumps([split_seed, list(transitions)], separators=(",", ":")).encode()
    bucket = int.from_bytes(hashlib.sha256(canonical).digest(), "big") % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def _permuted_index(index: int, size: int, key: bytes) -> int:
    """Six-round balanced Feistel permutation, cycle-walked into [0, size).

    This is a deterministic sampling permutation, not a cryptographic API. Its
    containing power-of-two domain is less than four times ``size``. Iterating
    distinct inputs therefore visits distinct table indices with O(1) state.
    """
    half_bits = max(1, ((size - 1).bit_length() + 1) // 2)
    mask = (1 << half_bits) - 1
    width = (half_bits + 7) // 8
    value = index
    while True:
        left, right = value >> half_bits, value & mask
        for round_index in range(6):
            digest = hashlib.shake_256(key + bytes([round_index]) + right.to_bytes(width, "big"))
            mixed = int.from_bytes(digest.digest(width), "big") & mask
            left, right = right, left ^ mixed
        value = (left << half_bits) | right
        if value < size:
            return value


class SampleStream:
    """Fresh tables from one split with a checkpointable local RNG.

    A stream cannot emit the same table twice. The finite table domain has N**N
    functions or (N-1)! directed cycles; requesting too many examples raises
    RuntimeError. Different stream seeds change visit order but do not change
    dataset ownership.
    """

    def __init__(
        self,
        config: DataConfig,
        split: str,
        seed: int,
        min_hops: int,
        max_hops: int,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if not 1 <= min_hops <= max_hops <= config.test_max_hops:
            raise ValueError("stream hop bounds must fit the configured range")
        self.config, self.split, self.seed = config, split, seed
        self.min_hops, self.max_hops = min_hops, max_hops
        self._rng = random.Random(seed)
        key_prefix = "pointer-table-permutation" if config.table_mode == "functional" else "pointer-cycle-permutation"
        self._key = hashlib.sha256(f"{key_prefix}:{seed}".encode()).digest()
        self._size = (config.num_objects ** config.num_objects if config.table_mode == "functional"
                      else math.factorial(config.num_objects - 1))
        self._cursor = 0
        self.count = 0

    def sample(self, hops: int | None = None) -> Example:
        if hops is not None and not self.min_hops <= hops <= self.max_hops:
            raise ValueError("requested hops outside stream bounds")
        while self._cursor < self._size:
            index = _permuted_index(self._cursor, self._size, self._key)
            self._cursor += 1
            if self.config.table_mode == "functional":
                destinations = []
                for _ in range(self.config.num_objects):
                    index, destination = divmod(index, self.config.num_objects)
                    destinations.append(destination)
                transitions = tuple(destinations)
            else:
                # Every directed Hamiltonian cycle has a unique representation
                # beginning at object 0. Unrank the remaining permutation and
                # construct successors directly; no arbitrary functions are
                # generated and discarded to obtain the cycle property.
                remaining = list(range(1, self.config.num_objects))
                cycle = [0]
                for width in range(len(remaining), 0, -1):
                    block = math.factorial(width - 1)
                    choice, index = divmod(index, block)
                    cycle.append(remaining.pop(choice))
                destinations = [0] * self.config.num_objects
                for source, destination in zip(cycle, cycle[1:] + cycle[:1]):
                    destinations[source] = destination
                transitions = tuple(destinations)
            if split_for_table(transitions, self.config.split_seed) == self.split:
                order = list(range(self.config.num_objects))
                self._rng.shuffle(order)
                example = Example(
                    transitions=transitions,
                    start=self._rng.randrange(self.config.num_objects),
                    hops=self._rng.randint(self.min_hops, self.max_hops) if hops is None else hops,
                    order=tuple(order),
                )
                self.count += 1
                return example
        raise RuntimeError(
            f"{self.split} stream exhausted its finite {self.config.table_mode} "
            f"table space after {self.count} examples; "
            "increase num_objects or reduce the requested sample count"
        )

    next_example = sample

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "config": asdict(self.config),
            "split": self.split,
            "seed": self.seed,
            "min_hops": self.min_hops,
            "max_hops": self.max_hops,
            "cursor": self._cursor,
            "count": self.count,
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = self.state_dict()
        # DataConfig defaults keep format-v1 functional-table checkpoints
        # resumable even though their serialized config lacks table_mode.
        state_config = asdict(DataConfig(**state["config"]))
        for field in ("version", "split", "seed", "min_hops", "max_hops"):
            if state[field] != expected[field]:
                raise ValueError(f"incompatible stream checkpoint: {field}")
        if state_config != expected["config"]:
            raise ValueError("incompatible stream checkpoint: config")
        cursor, count = int(state["cursor"]), int(state["count"])
        if not 0 <= count <= cursor <= self._size:
            raise ValueError("invalid stream checkpoint counters")
        self._rng.setstate(state["rng_state"])
        self._cursor, self.count = cursor, count


def make_eval_sets(config: DataConfig) -> dict[str, list[Example]]:
    """Fixed balanced hop strata; test_size examples per test condition.

    Counts for lengths within each condition differ by at most one. Validation
    is independent of both tests, and the shared test stream prevents any table
    repetition between the familiar-length and longer-length test conditions.
    """
    validation = SampleStream(
        config, "validation", config.split_seed + 1,
        config.train_min_hops, config.train_max_hops,
    )
    test = SampleStream(
        config, "test", config.split_seed + 2,
        config.train_min_hops, config.test_max_hops,
    )

    def generate(stream: SampleStream, size: int, minimum: int, maximum: int) -> list[Example]:
        return [stream.sample(hops=minimum + i % (maximum - minimum + 1)) for i in range(size)]

    return {
        "validation": generate(validation, config.val_size, config.train_min_hops, config.train_max_hops),
        "in_distribution": generate(test, config.test_size, config.train_min_hops, config.train_max_hops),
        "longer": generate(test, config.test_size, config.train_max_hops + 1, config.test_max_hops),
    }


def collate(examples: Sequence[Example], device: str | torch.device | None = None) -> dict[str, torch.Tensor]:
    """Right-pad conditions; targets are object-class indices, not token IDs."""
    if not examples:
        raise ValueError("cannot collate an empty batch")
    rows = [encode(example) for example in examples]
    max_length = max(map(len, rows))
    # Build complete tensors on the host so accelerator batches require only
    # one transfer per field, instead of per-example allocations and copies.
    padded_rows = [row + [PAD] * (max_length - len(row)) for row in rows]
    batch = {
        "input_ids": torch.tensor(padded_rows, dtype=torch.long, device="cpu"),
        "target": torch.tensor([example.target for example in examples], dtype=torch.long, device="cpu"),
        "hops": torch.tensor([example.hops for example in examples], dtype=torch.long, device="cpu"),
    }
    return batch if device is None else {name: value.to(device) for name, value in batch.items()}
