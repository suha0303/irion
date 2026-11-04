"""Layer-normalized forecast models for the SGD/Adam study."""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .legacy_models import ModelSpec as ModelSpec
from .legacy_models import make_model as make_legacy_model

STUDY_REVISION = "interlayer_norm_no_dropout"
NORM_EPS = 1e-5


@dataclass(frozen=True)
class StudySpec:
    name: str
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    revision: str = STUDY_REVISION

    def validate(self):
        if self.name not in {"lstm", "transformer"} or self.revision != STUDY_REVISION:
            raise ValueError("Unknown study model")
        if any(type(value) is not int for value in (self.hidden, self.layers, self.heads)):
            raise ValueError("Model dimensions must be integers")
        if self.hidden < 8 or not 1 <= self.layers <= 8:
            raise ValueError("Invalid model size")
        if self.heads < 1 or self.hidden % self.heads:
            raise ValueError("Hidden width must divide evenly into heads")
        return self

    def to_dict(self):
        return asdict(self)


class InterLayerNormStack(nn.Module):
    def __init__(self, inputs, spec):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.LSTM(inputs if i == 0 else spec.hidden, spec.hidden, batch_first=True)
            for i in range(spec.layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(spec.hidden, eps=NORM_EPS) for _ in range(spec.layers - 1)
        ])
        for layer in self.layers:
            for name, parameter in layer.named_parameters():
                if name.startswith("weight_ih"):
                    for gate in parameter.chunk(4, 0):
                        nn.init.xavier_uniform_(gate)
                elif name.startswith("weight_hh"):
                    for gate in parameter.chunk(4, 0):
                        nn.init.orthogonal_(gate)
                else:
                    nn.init.zeros_(parameter)
            # The two native biases are added. Initialize only one forget slice to+1.
            with torch.no_grad():
                layer.bias_ih_l0[spec.hidden:2 * spec.hidden].fill_(1)

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x, (hidden, _) = layer(x)
            if index < len(self.norms):
                x = self.norms[index](x)
        return x, hidden[-1]


class LSTMForecast(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec.validate()
        h = spec.hidden
        self.encoder = InterLayerNormStack(19, spec)
        self.fuse = nn.Sequential(nn.Linear(h + 6, h), nn.GELU())
        self.lead = nn.Embedding(10, h)
        self.decoder = InterLayerNormStack(2 * h, spec)
        self.head = nn.Linear(h, 4)
        for layer in (self.fuse[0], self.head):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.lead.weight, mean=0, std=1)
        self.register_buffer("lead_ids", torch.arange(10), persistent=False)

    def forward(self, history, context):
        _, hidden = self.encoder(history)
        latent = self.fuse(torch.cat([hidden, context], -1))
        lead = self.lead(self.lead_ids).unsqueeze(0).expand(len(history), -1, -1)
        x = torch.cat([latent.unsqueeze(1).expand(-1, 10, -1), lead], -1)
        output, _ = self.decoder(x)
        return self.head(output)


class PostEncoderLayer(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden)
        )
        self.norm1 = nn.LayerNorm(hidden, eps=NORM_EPS)
        self.norm2 = nn.LayerNorm(hidden, eps=NORM_EPS)

    def forward(self, x):
        attended = self.attention(x, x, x, need_weights=False)[0]
        x = self.norm1(x + attended)
        return self.norm2(x + self.feedforward(x))


class PostDecoderLayer(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden)
        )
        self.norm1 = nn.LayerNorm(hidden, eps=NORM_EPS)
        self.norm2 = nn.LayerNorm(hidden, eps=NORM_EPS)
        self.norm3 = nn.LayerNorm(hidden, eps=NORM_EPS)

    def forward(self, x, memory):
        attended = self.self_attention(x, x, x, need_weights=False)[0]
        x = self.norm1(x + attended)
        attended = self.cross_attention(x, memory, memory, need_weights=False)[0]
        x = self.norm2(x + attended)
        return self.norm3(x + self.feedforward(x))


class TransformerForecast(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec.validate()
        h = spec.hidden
        self.project, self.context = nn.Linear(19, h), nn.Linear(6, h)
        self.encoder = nn.ModuleList([PostEncoderLayer(h, spec.heads) for _ in range(spec.layers)])
        self.decoder = nn.ModuleList([PostDecoderLayer(h, spec.heads) for _ in range(spec.layers)])
        self.queries, self.head = nn.Embedding(10, h), nn.Linear(h, 4)
        for name, parameter in self.named_parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
            elif name.endswith("weight"):
                nn.init.ones_(parameter)
            else:
                nn.init.zeros_(parameter)
        position = torch.arange(100).unsqueeze(1)
        div = torch.exp(torch.arange(0, h, 2) * (-math.log(10000.0) / h))
        pe = torch.zeros(100, h)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div), torch.cos(position * div[:h // 2])
        self.register_buffer("position_encoding", pe, persistent=False)
        self.register_buffer("query_ids", torch.arange(10), persistent=False)

    def forward(self, history, context):
        if history.shape[1:] != (100, 19):
            raise ValueError("Fixed history100/features19 required")
        encoded = self.project(history) + self.position_encoding
        memory = torch.cat([encoded, self.context(context).unsqueeze(1)], 1)
        for layer in self.encoder:
            memory = layer(memory)
        queries = self.queries(self.query_ids).unsqueeze(0).expand(len(history), -1, -1)
        for layer in self.decoder:
            queries = layer(queries, memory)
        return self.head(queries)


def make_model(spec):
    if spec.revision == "explicit_dropout_v1":
        return make_legacy_model(spec)
    spec.validate()
    return (LSTMForecast if spec.name == "lstm" else TransformerForecast)(spec)
