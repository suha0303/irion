import math

import torch
from torch import nn


class LSTMForecast(nn.Module):
    def __init__(self, features=19, context=6, horizon=10, hidden=32):
        super().__init__()
        self.encoder = nn.LSTM(features, hidden, 2, batch_first=True, dropout=0.1)
        self.fuse = nn.Sequential(nn.Linear(hidden + context, hidden), nn.GELU())
        self.lead = nn.Embedding(horizon, hidden)
        self.decoder = nn.LSTM(hidden * 2, hidden, 2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, 4)
        self.horizon = horizon

    def forward(self, history, context):
        _, (h, _) = self.encoder(history)
        latent = self.fuse(torch.cat([h[-1], context], dim=-1))
        lead = (
            self.lead(torch.arange(self.horizon, device=history.device))
            .unsqueeze(0)
            .expand(len(history), -1, -1)
        )
        decoded, _ = self.decoder(
            torch.cat([latent.unsqueeze(1).expand(-1, self.horizon, -1), lead], dim=-1)
        )
        return self.head(decoded)


class TransformerForecast(nn.Module):
    def __init__(self, features=19, context=6, horizon=10, hidden=32):
        super().__init__()
        self.project = nn.Linear(features, hidden)
        self.context = nn.Linear(context, hidden)
        layer = nn.TransformerEncoderLayer(
            hidden, 4, hidden * 2, 0.1, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        decoder_layer = nn.TransformerDecoderLayer(
            hidden, 4, hidden * 2, 0.1, activation="gelu", batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, 2)
        self.queries = nn.Embedding(horizon, hidden)
        self.head = nn.Linear(hidden, 4)
        self.horizon, self.hidden = horizon, hidden
        # Cloned layers must not retain identical initial weights.
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, history, context):
        length = history.shape[1]
        pos = torch.arange(length, device=history.device).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.hidden, 2, device=history.device)
            * (-math.log(10000.0) / self.hidden)
        )
        pe = torch.zeros(length, self.hidden, device=history.device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        memory = self.encoder(
            torch.cat([self.project(history) + pe, self.context(context).unsqueeze(1)], dim=1)
        )
        query = (
            self.queries(torch.arange(self.horizon, device=history.device))
            .unsqueeze(0)
            .expand(len(history), -1, -1)
        )
        return self.head(self.decoder(query, memory))


def make_model(name, hidden=32):
    if name == "lstm":
        return LSTMForecast(hidden=hidden)
    if name == "transformer":
        return TransformerForecast(hidden=hidden)
    raise ValueError("unknown model " + name)


def device_for(requested):
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    return torch.device(requested)
