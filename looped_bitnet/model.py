"""A fixed input memory and one recurrent working-state vector.

Input grammar (right padded): [EDGE, OBJ(src), OBJ(dst)] * num_objects,
[START, OBJ(start)], [STEP] * hops, [END]. Object IDs are offset by 5.
The encoder treats each edge separately: no sequence Transformer or cross-edge
mixing can solve the task ahead of the recurrent reader. Unary STEP tokens are
counted and represented by a single learned direction, with no unseen length
tokens or positional embeddings. The count never controls the recurrent loop.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .data import OBJ_OFFSET, STEP
from .quantization import make_linear


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    d_ff: int = 1024
    num_blocks: int = 4
    num_heads: int = 4
    num_objects: int = 16
    max_tokens: int = 256
    steps: int = 8
    quantized: bool = True
    persistent_query: bool = False
    structured_reader: bool = False

    def __post_init__(self) -> None:
        for name in ("d_model", "d_ff", "num_blocks", "num_heads", "num_objects", "max_tokens", "steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.max_tokens < 3 * self.num_objects + 3:
            raise ValueError("max_tokens cannot fit all edges, START/object, and END")
        if not isinstance(self.quantized, bool):
            raise ValueError("quantized must be a bool")
        if not isinstance(self.persistent_query, bool):
            raise ValueError("persistent_query must be a bool")
        if not isinstance(self.structured_reader, bool):
            raise ValueError("structured_reader must be a bool")


class AttentionReader(nn.Module):
    """Shared reader: K/V are created once, then used by every recurrent step."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.q_norm = nn.LayerNorm(config.d_model, dtype=torch.float32)
        self.q_proj = make_linear(config.d_model, config.d_model, quantized=config.quantized)
        self.k_proj = make_linear(config.d_model, config.d_model, quantized=config.quantized)
        self.v_proj = make_linear(config.d_model, config.d_model, quantized=config.quantized)
        self.out_proj = make_linear(config.d_model, config.d_model, quantized=config.quantized)

    def build_kv(self, key_memory: Tensor, value_memory: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Project one aligned memory table into cached keys and values.

        With one argument this is the legacy mixed-memory reader. Passing a
        second table keeps rows aligned while allowing a structured reader to
        source keys and values from separate representations.
        """
        memory = key_memory
        if value_memory is None:
            value_memory = memory
        batch, edges, _ = memory.shape
        # No detach: all uses backpropagate through the original memory graph.
        keys = self.k_proj(memory).view(batch, edges, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(value_memory).view(batch, edges, self.num_heads, self.head_dim).transpose(1, 2)
        return keys, values

    def forward(self, state: Tensor, cache: tuple[Tensor, Tensor]) -> Tensor:
        keys, values = cache
        query = self.q_proj(self.q_norm(state)).view(-1, self.num_heads, 1, self.head_dim)
        # Explicit FP32 score, softmax, and value accumulation for numerical
        # clarity. We intentionally do not use low precision fused attention.
        scores = torch.matmul(query.float(), keys.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, values.float()).reshape(state.shape[0], -1)
        return self.out_proj(context)


class FFNBlock(nn.Module):
    """One independent pre-norm FFN, shared with itself on later visits."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model, dtype=torch.float32)
        self.up = make_linear(config.d_model, config.d_ff, quantized=config.quantized)
        self.activation = nn.GELU()
        self.down = make_linear(config.d_ff, config.d_model, quantized=config.quantized)

    def forward(self, state: Tensor) -> Tensor:
        return state + self.down(self.activation(self.up(self.norm(state))))


class ReasoningModel(nn.Module):
    """Cyclic recurrent core with one working vector h per example.

    Each step reads a fixed table memory, adds the readout to h, then applies
    the selected residual FFN. A final LayerNorm and one small linear map
    predict an object. Keep parameters in FP32 and move only their device.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.source_embedding = nn.Embedding(config.num_objects, config.d_model, dtype=torch.float32)
        self.destination_embedding = nn.Embedding(config.num_objects, config.d_model, dtype=torch.float32)
        self.start_embedding = nn.Embedding(config.num_objects, config.d_model, dtype=torch.float32)
        self.memory_norm = nn.LayerNorm(config.d_model, dtype=torch.float32)
        self.step_embedding = nn.Parameter(torch.empty(config.d_model, dtype=torch.float32))
        self.reader = AttentionReader(config)
        self.query_adapter = (make_linear(2 * config.d_model, config.d_model,
                                          quantized=config.quantized)
                              if config.persistent_query else None)
        self.blocks = nn.ModuleList(FFNBlock(config) for _ in range(config.num_blocks))
        self.output_norm = nn.LayerNorm(config.d_model, dtype=torch.float32)
        self.output_head = make_linear(config.d_model, config.num_objects, quantized=config.quantized)
        self.apply(self._initialize)
        # Match the object embedding scale so the unary count does not drown
        # out the identity of the starting object at initialization.
        nn.init.normal_(self.step_embedding, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def select_block(self, step: int) -> int:
        """A deliberately small routing boundary, replaceable in later stages."""
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        return step % len(self.blocks)

    def encode_input(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Create fixed edge memory and initial h without cross-edge mixing.

        Count encoding is h0 = start_embedding + hops * step_embedding /
        sqrt(d_model). This linear rule works for new counts with the same
        STEP token, but successful length extrapolation remains an experiment.
        """
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise ValueError("input_ids must be a torch.long tensor of shape [batch, tokens]")
        edge_end = 3 * self.config.num_objects
        if input_ids.shape[0] == 0:
            raise ValueError("input_ids must contain at least one example")
        if not edge_end + 3 <= input_ids.shape[1] <= self.config.max_tokens:
            raise ValueError("input length does not fit the model's edge table or max_tokens")
        edges = input_ids[:, :edge_end].reshape(-1, self.config.num_objects, 3)
        source = edges[:, :, 1] - OBJ_OFFSET
        destination = edges[:, :, 2] - OBJ_OFFSET
        start = input_ids[:, edge_end + 1] - OBJ_OFFSET
        # Embedding indexing checks that object IDs are in range. The generator
        # and tokenizer own full grammar validation; avoid per-step host syncs.
        memory = self.memory_norm(self.source_embedding(source) + self.destination_embedding(destination))
        hops = (input_ids[:, edge_end + 2:] == STEP).sum(dim=-1, keepdim=True).float()
        state = self.start_embedding(start) + hops * self.step_embedding / math.sqrt(self.config.d_model)
        return memory, state

    def _encode_structured_input(self, input_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode aligned source and destination rows for the structured reader."""
        if input_ids.dtype != torch.long or input_ids.ndim != 2:
            raise ValueError("input_ids must be a torch.long tensor of shape [batch, tokens]")
        edge_end = 3 * self.config.num_objects
        if input_ids.shape[0] == 0:
            raise ValueError("input_ids must contain at least one example")
        if not edge_end + 3 <= input_ids.shape[1] <= self.config.max_tokens:
            raise ValueError("input length does not fit the model's edge table or max_tokens")
        edges = input_ids[:, :edge_end].reshape(-1, self.config.num_objects, 3)
        source = edges[:, :, 1] - OBJ_OFFSET
        destination = edges[:, :, 2] - OBJ_OFFSET
        start = input_ids[:, edge_end + 1] - OBJ_OFFSET
        source_memory = self.memory_norm(self.source_embedding(source))
        destination_memory = self.memory_norm(self.destination_embedding(destination))
        hops = (input_ids[:, edge_end + 2:] == STEP).sum(dim=-1, keepdim=True).float()
        state = self.start_embedding(start) + hops * self.step_embedding / math.sqrt(self.config.d_model)
        return source_memory, destination_memory, state

    def _run(self, input_ids: Tensor, steps: int | None, collect_diagnostics: bool,
             readout_steps: tuple[int, ...] | None = None):
        budget = self.config.steps if steps is None else steps
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError("steps must be a positive integer")
        if readout_steps is not None:
            if collect_diagnostics:
                raise ValueError("cannot collect diagnostics and readouts together")
            if (not readout_steps or any(isinstance(step, bool) or not isinstance(step, int) or step <= 0
                                         or step > budget for step in readout_steps)
                    or len(set(readout_steps)) != len(readout_steps)):
                raise ValueError("readout_steps must contain distinct positive steps within the budget")
            requested_readouts = set(readout_steps)
        else:
            requested_readouts = set()
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            if self.config.structured_reader:
                key_memory, value_memory, state = self._encode_structured_input(input_ids)
                cache = self.reader.build_kv(key_memory, value_memory)
            else:
                memory, state = self.encode_input(input_ids)
                cache = self.reader.build_kv(memory)
            original_query = state
            diagnostics = []
            readouts = {}
            for step in range(budget):
                before = state
                reader_input = (self.query_adapter(torch.cat((state, original_query), dim=-1))
                                if self.query_adapter is not None else state)
                reader_update = self.reader(reader_input, cache)
                after_reader = state + reader_update
                state = self.blocks[self.select_block(step)](after_reader)
                if step + 1 in requested_readouts:
                    # This is the same shared readout used for the final answer;
                    # no extra parameters or target-dependent path is involved.
                    readouts[step + 1] = self.output_head(self.output_norm(state))
                if collect_diagnostics:
                    diagnostics.append({
                        "step": step + 1,
                        "state_before_l2": before.float().norm(dim=-1).mean().item(),
                        "reader_update_l2": reader_update.float().norm(dim=-1).mean().item(),
                        "state_after_reader_l2": after_reader.float().norm(dim=-1).mean().item(),
                        "ffn_update_l2": (state - after_reader).float().norm(dim=-1).mean().item(),
                        "state_after_l2": state.float().norm(dim=-1).mean().item(),
                    })
            logits = self.output_head(self.output_norm(state))
            if readout_steps is not None:
                return logits, readouts
            return (logits, diagnostics) if collect_diagnostics else logits

    def forward(self, input_ids: Tensor, steps: int | None = None) -> Tensor:
        return self._run(input_ids, steps, False)

    def forward_diagnostics(self, input_ids: Tensor, steps: int | None = None):
        """Return logits and read-only per-step norm summaries for evaluation."""
        return self._run(input_ids, steps, True)

    def forward_with_readouts(self, input_ids: Tensor, readout_steps: tuple[int, ...] = (4,),
                              steps: int | None = None):
        """Return final logits plus shared-head logits at selected recurrent steps.

        Readouts are an optional training/evaluation diagnostic. They do not
        change the default ``forward`` output and never feed predictions back
        into the recurrent state.
        """
        return self._run(input_ids, steps, False, tuple(readout_steps))
