"""Versioned forecast models with explicit, checkpointed PyTorch dropout randomness."""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    revision: str = "explicit_dropout_v1"

    def validate(self):
        if self.name not in {"lstm", "transformer"} or self.revision != "explicit_dropout_v1":
            raise ValueError("Unknown model/revision")
        if self.hidden < 8 or not 1 <= self.layers <= 8 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid model size/dropout")
        if self.heads < 1 or self.hidden % self.heads:
            raise ValueError("Hidden width must divide evenly into heads")
        return self

    def to_dict(self):
        return asdict(self)


class ExplicitDropoutStack(nn.Module):
    def __init__(self, inputs, spec):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.LSTM(
                    inputs if i == 0 else spec.hidden, spec.hidden, 1, batch_first=True, dropout=0
                )
                for i in range(spec.layers)
            ]
        )
        self.dropout = nn.Dropout(spec.dropout)

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x, (hidden, _) = layer(x)
            if index + 1 < len(self.layers):
                x = self.dropout(x)
        return x, hidden[-1]


class LSTMForecast(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec.validate()
        h = spec.hidden
        self.encoder = ExplicitDropoutStack(19, spec)
        self.fuse = nn.Sequential(nn.Linear(h + 6, h), nn.GELU())
        self.lead = nn.Embedding(10, h)
        self.decoder = ExplicitDropoutStack(2 * h, spec)
        self.head = nn.Linear(h, 4)
        self.register_buffer("lead_ids", torch.arange(10), persistent=False)

    def forward(self, history, context):
        _, hidden = self.encoder(history)
        latent = self.fuse(torch.cat([hidden, context], -1))
        lead = self.lead(self.lead_ids).unsqueeze(0).expand(len(history), -1, -1)
        x = torch.cat([latent.unsqueeze(1).expand(-1, 10, -1), lead], -1)
        output, _ = self.decoder(x)
        return self.head(output)


class TransformerForecast(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec.validate()
        h = spec.hidden
        self.project, self.context = nn.Linear(19, h), nn.Linear(6, h)
        encoder = nn.TransformerEncoderLayer(
            h, spec.heads, 2 * h, spec.dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder, spec.layers, enable_nested_tensor=False)
        decoder = nn.TransformerDecoderLayer(
            h, spec.heads, 2 * h, spec.dropout, activation="gelu", batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder, spec.layers)
        self.queries, self.head = nn.Embedding(10, h), nn.Linear(h, 4)
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        # Fixed history length; avoid rebuilding deterministic positional tensors each batch.
        position = torch.arange(100).unsqueeze(1)
        div = torch.exp(torch.arange(0, h, 2) * (-math.log(10000.0) / h))
        pe = torch.zeros(100, h)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div), torch.cos(position * div[: h // 2])
        self.register_buffer("position_encoding", pe, persistent=False)
        self.register_buffer("query_ids", torch.arange(10), persistent=False)

    def forward(self, history, context):
        if history.shape[1:] != (100, 19):
            raise ValueError("Fixed history100/features19 required")
        encoded = self.project(history) + self.position_encoding
        memory = self.encoder(torch.cat([encoded, self.context(context).unsqueeze(1)], 1))
        queries = self.queries(self.query_ids).unsqueeze(0).expand(len(history), -1, -1)
        return self.head(self.decoder(queries, memory))


def make_model(spec):
    return (LSTMForecast if spec.validate().name == "lstm" else TransformerForecast)(spec)
