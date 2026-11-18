"""Fixed forecast and whole-run variants for the new common19 experiment."""

import math
from dataclasses import dataclass

import torch
from torch import nn

HISTORY, HORIZON, FEATURES = 100, 10, 19
PATCH, STRIDE, VOLTAGE = 100, 50, 17
DROPOUT = 0.1


@dataclass(frozen=True)
class Variant:
    task: str
    architecture: str
    hidden: int
    layers: int

    @property
    def name(self):
        return f"{self.task}-{self.architecture}-{self.hidden}x{self.layers}"


VARIANTS = {
    variant.name: variant
    for task in ("inflight", "whole-run")
    for architecture in ("lstm", "transformer")
    for hidden, layers in ((64, 2), (128, 4))
    for variant in (Variant(task, architecture, hidden, layers),)
}


def past_mask(length, device):
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)


def positions(sequence):
    size, hidden = sequence.shape[-2:]
    time = torch.arange(size, device=sequence.device, dtype=sequence.dtype)
    rate = torch.arange(0, hidden, 2, device=time.device, dtype=time.dtype)
    phase = time[:, None] * torch.exp(rate * (-math.log(10000.0) / hidden))
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)


def valid_steps(lengths, size):
    return torch.arange(size, device=lengths.device)[None] < lengths[:, None]


class SequenceEncoder(nn.Module):
    def __init__(self, width, variant):
        super().__init__()
        h = variant.hidden
        self.project = nn.Linear(width, h)
        self.architecture = variant.architecture
        if self.architecture == "lstm":
            # With right padding, a unidirectional LSTM's valid outputs are
            # independent of padding. Gather the last valid step explicitly.
            self.layers = nn.ModuleList(
                nn.LSTM(h, h, batch_first=True) for _ in range(variant.layers)
            )
            self.norms = nn.ModuleList(
                nn.LayerNorm(h) for _ in range(variant.layers - 1)
            )
            self.dropout = nn.Dropout(DROPOUT)
        else:
            self.layers = nn.ModuleList(
                nn.TransformerEncoderLayer(
                    h,
                    4,
                    2 * h,
                    DROPOUT,
                    "gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(variant.layers)
            )
            self.final_norm = nn.LayerNorm(h)

    def forward(self, x, lengths=None, *, causal=False):
        x = self.project(x)
        if self.architecture == "lstm":
            for i, layer in enumerate(self.layers):
                x, _ = layer(x)
                if i < len(self.norms):
                    x = self.dropout(self.norms[i](x))
            return x
        x = x + positions(x)
        padding = None if lengths is None else ~valid_steps(lengths, x.shape[1])
        mask = past_mask(x.shape[1], x.device) if causal else None
        for layer in self.layers:
            x = layer(x, src_mask=mask, src_key_padding_mask=padding)
        return self.final_norm(x)


class DeltaForecast(nn.Module):
    """Output ten deltas anchored to one origin, with no target-side input."""

    def __init__(self, variant):
        super().__init__()
        h = variant.hidden
        self.variant = variant
        self.encoder = SequenceEncoder(FEATURES, variant)
        self.leads = nn.Parameter(torch.empty(HORIZON, h))
        nn.init.normal_(self.leads, std=0.02)
        if variant.architecture == "lstm":
            self.decoder = SequenceEncoder(2 * h, variant)
        else:
            self.decoder = nn.ModuleList(
                nn.TransformerDecoderLayer(
                    h,
                    4,
                    2 * h,
                    DROPOUT,
                    "gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(variant.layers)
            )
            self.decoder_norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, 4)

    def forward(self, history):
        if history.ndim != 3 or history.shape[1:] != (HISTORY, FEATURES):
            raise ValueError("Expected history with shape [B,100,19]")
        if not len(history) or not bool(torch.isfinite(history).all()):
            raise ValueError("History must contain finite observations")
        memory = self.encoder(history)
        future = self.leads.unsqueeze(0).expand(len(history), -1, -1)
        if self.variant.architecture == "lstm":
            state = memory[:, -1:, :].expand(-1, HORIZON, -1)
            future = self.decoder(torch.cat((state, future), dim=-1))
        else:
            mask = past_mask(HORIZON, history.device)
            for layer in self.decoder:
                future = layer(future, memory, tgt_mask=mask)
            future = self.decoder_norm(future)
        return self.head(future)


class RunVoltage(nn.Module):
    """Construct one normal voltage reference from overlapping patches."""

    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        h = variant.hidden
        self.local = SequenceEncoder(FEATURES + 1, variant)
        self.run = SequenceEncoder(h, variant)
        self.initial = nn.Linear(h, h)
        self.head = nn.Sequential(
            nn.Linear(2 * h + FEATURES + 1, h), nn.GELU(), nn.Linear(h, 1)
        )

    def forward(self, values, lengths):
        if values.ndim != 3 or values.shape[2] != FEATURES or not len(values):
            raise ValueError(
                "Expected whole-run conditioning with shape [B,T,19]"
            )
        batch, time, _ = values.shape
        if (
            lengths.shape != (batch,)
            or lengths.dtype != torch.int64
            or lengths.device != values.device
        ):
            raise ValueError(
                "Expected one int64 length per run on the input device"
            )
        if bool(((lengths <= PATCH) | (lengths > time)).any()):
            raise ValueError(
                "Each run needs data after the initial ten seconds"
            )
        valid = valid_steps(lengths, time)
        initial_values = values[:, :PATCH]
        initial = torch.cat(
            (initial_values, torch.ones_like(initial_values[..., :1])), dim=-1
        )
        # Only the initial branch can use measured voltage. It is removed
        # before the finite check so missing post-initial labels cannot leak.
        reference = torch.cat(
            (
                values[..., :VOLTAGE],
                torch.zeros_like(values[..., :1]),
                values[..., VOLTAGE + 1 :],
                torch.zeros_like(values[..., :1]),
            ),
            dim=-1,
        ).masked_fill(~valid[..., None], 0)
        if not bool(
            torch.isfinite(initial).all() and torch.isfinite(reference).all()
        ):
            raise ValueError(
                "Missing conditioning features in the admitted segment"
            )
        initial_state = self.initial(self.local(initial)[:, -1])

        counts = 1 + (lengths - PATCH + STRIDE - 1) // STRIDE
        n_patches = 1 + (time - PATCH + STRIDE - 1) // STRIDE
        starts = torch.arange(n_patches, device=values.device) * STRIDE
        indices = starts[:, None] + torch.arange(PATCH, device=values.device)
        patch_lengths = (lengths[:, None] - starts).clamp(0, PATCH)
        active = valid_steps(counts, n_patches)
        frames = reference[:, indices.clamp(max=time - 1)]
        patch_valid = valid_steps(patch_lengths[active], PATCH)
        selected = frames[active].masked_fill(~patch_valid[..., None], 0)
        local = self.local(selected, patch_lengths[active])
        final_step = patch_lengths[active] - 1
        latent = values.new_zeros((batch, n_patches, self.variant.hidden))
        latent[active] = local[
            torch.arange(len(local), device=values.device), final_step
        ]
        context = self.run(latent + initial_state[:, None], counts, causal=True)
        decoded = self.head(
            torch.cat(
                (
                    local,
                    context[active][:, None].expand(-1, PATCH, -1),
                    selected,
                ),
                dim=-1,
            )
        ).squeeze(-1)

        # A fixed slice addition per patch is deterministic on CPU and CUDA.
        # It also avoids counting overlap twice in the loss.
        predictions = values.new_zeros((batch, n_patches, PATCH))
        predictions[active] = decoded.masked_fill(~patch_valid, 0)
        aligned = []
        aligned_counts = []
        # Even patches never overlap each other, nor do odd patches.
        for parity in (0, 1):
            start = parity * STRIDE
            chunks = predictions[:, parity::2].flatten(1)
            present = (
                indices[None, parity::2] < lengths[:, None, None]
            ) & active[:, parity::2, None]
            aligned.append(
                nn.functional.pad(
                    chunks, (start, max(0, time - start - chunks.shape[1]))
                )[:, :time]
            )
            aligned_counts.append(
                nn.functional.pad(
                    present.flatten(1),
                    (start, max(0, time - start - chunks.shape[1])),
                )[:, :time]
            )
        total = aligned[0] + aligned[1]
        count = aligned_counts[0].to(values.dtype) + aligned_counts[1].to(
            values.dtype
        )
        return (total / count.clamp_min(1)).unsqueeze(-1)


def make_model(name):
    if name not in VARIANTS:
        raise ValueError(f"Unknown new19 model: {name}")
    variant = VARIANTS[name]
    return (DeltaForecast if variant.task == "inflight" else RunVoltage)(
        variant
    )
