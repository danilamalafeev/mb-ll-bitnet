"""BitLinear-like fake quantization; tensors and master parameters stay FP32.

Weights use an absmean scale and ternary levels, and activations use symmetric
per-vector INT8 levels. These operators simulate quantization in a normal
floating-point linear operation. They do not pack weights or invoke low-bit
kernels, and therefore make no storage or performance savings claim.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def ternary_weight(weight: Tensor, eps: float = 1e-8) -> Tensor:
    """Return dequantized {-scale, 0, scale} weights with an identity STE.

    The scale is the mean absolute weight over this projection's entire matrix.
    Detaching the quantization correction gives d(output)/d(weight) = 1 rather
    than differentiating through round, clipping, or the estimated scale.
    """
    weight = weight.float()
    scale = weight.abs().mean().clamp_min(eps)
    quantized = (weight / scale).round().clamp(-1, 1) * scale
    return weight + (quantized - weight).detach()


def int8_activation(value: Tensor, eps: float = 1e-8) -> Tensor:
    """Symmetric fake INT8 quantization per last-dimension vector, with STE.

    Each vector has its own absmax/127 scale. The symmetric integer range is
    [-127, 127]; -128 is deliberately unused. A positive scale floor makes
    all-zero vectors finite while preserving them exactly.
    """
    value = value.float()
    scale = (value.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(eps)
    quantized = (value / scale).round().clamp(-127, 127) * scale
    return value + (quantized - value).detach()


# Descriptive aliases are convenient for callers inspecting the QAT operators.
weight_fake_quant = ternary_weight
activation_fake_quant = int8_activation


class BitLinear(nn.Linear):
    """A linear layer with FP32 trainable masters and forward fake quantization.

    Normalization belongs to the enclosing pre-norm architecture, rather than
    being hidden inside every projection. Bias, if enabled, remains FP32.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias, dtype=torch.float32)

    def forward(self, value: Tensor) -> Tensor:
        with torch.autocast(device_type=value.device.type, enabled=False):
            bias = self.bias.float() if self.bias is not None else None
            return F.linear(int8_activation(value), ternary_weight(self.weight), bias)


def make_linear(in_features: int, out_features: int, *, quantized: bool) -> nn.Linear:
    """Use the same bias-free architecture for QAT and full precision controls."""
    if quantized:
        return BitLinear(in_features, out_features)
    return nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
