"""Shared offline run hierarchy with the same eight sizes as history19."""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from learning.history19.models import InterLayerNormStack

REVISION = "run-health19-voltage-reference/1"
BLOCK = 100  # 10 Hz observations, non-overlapping ten-second blocks.
VOLTAGE = 17


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    revision: str = REVISION

    def validate(self):
        if self.name not in {"lstm", "transformer"} or self.revision != REVISION:
            raise ValueError("Unknown run-health model")
        if any(type(v) is not int for v in (self.hidden, self.layers, self.heads)):
            raise ValueError("Integer model dimensions required")
        if self.hidden not in {64, 128} or self.layers not in {2, 4} or self.heads != 4:
            raise ValueError("This experiment selects hidden64/128, layers2/4, heads4")
        return self

    def to_dict(self):
        return asdict(self)


def position(size, hidden, reference):
    time = torch.arange(size, device=reference.device, dtype=reference.dtype)[:, None]
    rate = torch.exp(
        torch.arange(0, hidden, 2, device=reference.device, dtype=reference.dtype)
        * (-math.log(10000.0) / hidden)
    )
    value = reference.new_zeros(size, hidden)
    value[:, 0::2], value[:, 1::2] = torch.sin(time * rate), torch.cos(time * rate)
    return value


class Encoder(nn.Module):
    def __init__(self, inputs, spec):
        super().__init__()
        self.name = spec.name
        if self.name == "lstm":
            self.stack = InterLayerNormStack(inputs, spec)
        else:
            self.project = nn.Linear(inputs, spec.hidden)
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        spec.hidden,
                        spec.heads,
                        2 * spec.hidden,
                        dropout=0,
                        activation="gelu",
                        batch_first=True,
                        norm_first=False,
                        layer_norm_eps=1e-5,
                    )
                    for _ in range(spec.layers)
                ]
            )
            for parameter in self.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
            # Keep LayerNorm weight=1, bias=0; zero linear/MHA biases.
            for name, parameter in self.named_parameters():
                if "bias" in name:
                    nn.init.zeros_(parameter)

    def forward(self, x, lengths, *, causal=False):
        if self.name == "lstm":
            for index, layer in enumerate(self.stack.layers):
                packed = pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                packed, _ = layer(packed)
                x, _ = pad_packed_sequence(packed, batch_first=True, total_length=x.shape[1])
                if index < len(self.stack.norms):
                    x = self.stack.norms[index](x)
        else:
            x = self.project(x)
            x = x + position(x.shape[1], x.shape[2], x)
            padding = torch.arange(x.shape[1], device=x.device)[None] >= lengths[:, None]
            mask = (
                torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).triu(1)
                if causal
                else None
            )
            for layer in self.layers:
                x = layer(x, src_mask=mask, src_key_padding_mask=padding)
        last = x[torch.arange(len(x), device=x.device), lengths - 1]
        return x, last


class RunVoltageReference(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec.validate()
        h = spec.hidden
        self.local = Encoder(19, spec)
        self.initial = nn.Sequential(nn.Linear(h, h), nn.GELU())
        self.run_encoder = Encoder(h, spec)
        # Retain within-block dynamics as well as the compressed long-run state.
        self.head = nn.Sequential(nn.Linear(2 * h + 19, h), nn.GELU(), nn.Linear(h, 1))
        for module in (self.initial, self.head):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, values, lengths):
        if values.ndim != 3 or values.shape[-1] != 19 or len(values) == 0:
            raise ValueError("Expected run values[B,T,19]")
        if values.dtype != torch.float32 or lengths.dtype != torch.int64:
            raise ValueError("FP32 observations and int64 lengths required")
        if lengths.shape != (len(values),) or lengths.device != values.device:
            raise ValueError("One length per run on the same device required")
        if bool(((lengths < 1) | (lengths > values.shape[1])).any()):
            raise ValueError("Invalid run lengths")
        valid = torch.arange(values.shape[1], device=values.device)[None] < lengths[:, None]
        if not bool(torch.isfinite(values[valid]).all()):
            raise ValueError("Missing/nonfinite valid observations cannot be zero-filled")
        # Ignore padding before any attention or arithmetic, including NaN padding.
        x = values.masked_fill(~valid[:, :, None], 0)
        initial_length = lengths.clamp(max=BLOCK)
        _, initial = self.local(x[:, :BLOCK], initial_length)
        initial = self.initial(initial)
        reference = x.clone()
        reference[:, :, VOLTAGE] = 0  # No measured voltage after initialization in reference.
        blocks = (values.shape[1] + BLOCK - 1) // BLOCK
        padded = torch.nn.functional.pad(reference, (0, 0, 0, blocks * BLOCK - x.shape[1]))
        block_lengths = (
            lengths[:, None] - BLOCK * torch.arange(blocks, device=x.device)[None]
        ).clamp(min=0, max=BLOCK)
        active = block_lengths.flatten() > 0
        local_states, encoded = self.local(
            padded.reshape(-1, BLOCK, 19)[active], block_lengths.flatten()[active]
        )
        latent = x.new_zeros(len(x) * blocks, self.spec.hidden)
        latent[active] = encoded
        local = x.new_zeros(len(x) * blocks, BLOCK, self.spec.hidden)
        local[active] = local_states
        local = local.reshape(len(x), blocks * BLOCK, self.spec.hidden)[:, : x.shape[1]]
        latent = latent.reshape(len(x), blocks, self.spec.hidden) + initial[:, None]
        states, _ = self.run_encoder(latent, (lengths + BLOCK - 1) // BLOCK, causal=True)
        per_time = states.repeat_interleave(BLOCK, dim=1)[:, : x.shape[1]]
        predicted = self.head(torch.cat([per_time, local, reference], dim=-1)).squeeze(-1)
        return predicted.masked_fill(~valid, 0)


def make_model(spec):
    return RunVoltageReference(spec)


def scoring_mask(lengths, time):
    indices = torch.arange(time, device=lengths.device)[None]
    return (indices >= BLOCK) & (indices < lengths[:, None])


def normal_loss(prediction, observed, lengths):
    mask = scoring_mask(lengths, prediction.shape[1])
    counts = mask.sum(1)
    if bool((counts == 0).any()):
        raise ValueError("A run needs observations after the initial ten seconds")
    errors = (prediction - observed).masked_fill(~mask, 0).square()
    # Equal weight per run; longer flights must not silently dominate training.
    return (errors.sum(1) / counts).mean()
