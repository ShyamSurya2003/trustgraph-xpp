import torch
import torch.nn as nn
import torch.nn.functional as F


class TabTransformerLite(nn.Module):
    def __init__(self, n_features: int, d_model: int = 32):
        super().__init__()
        self.feature_tokens = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.value_proj = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=96, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(n_features * d_model), nn.Linear(n_features * d_model, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        tokens = self.value_proj(x.unsqueeze(-1)) + self.feature_tokens.unsqueeze(0)
        encoded = self.encoder(tokens).flatten(1)
        return self.head(encoded).squeeze(-1)


try:
    from torch_geometric.nn import GATConv
except Exception:  # pragma: no cover
    GATConv = None


class GATRiskModel(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 24):
        super().__init__()
        if GATConv is None:
            self.fallback = nn.Sequential(nn.Linear(in_channels, hidden), nn.ELU(), nn.Linear(hidden, 2))
        else:
            self.gat1 = GATConv(in_channels, hidden, heads=2, dropout=0.15)
            self.gat2 = GATConv(hidden * 2, 2, heads=1, concat=False, dropout=0.15)

    def forward(self, x, edge_index):
        if GATConv is None:
            return self.fallback(x)
        x = F.elu(self.gat1(x, edge_index))
        return self.gat2(x, edge_index)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 28, latent: int = 12):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        z = torch.tanh(self.to_latent(h[-1]))
        h0 = self.from_latent(z).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        seed = torch.zeros_like(x)
        decoded, _ = self.decoder(seed, (h0, c0))
        return self.output(decoded)


class AttentionFusionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 3), nn.Softmax(dim=-1))
        self.mlp = nn.Sequential(nn.Linear(3, 24), nn.ReLU(), nn.Linear(24, 1), nn.Sigmoid())

    def forward(self, scores):
        weights = self.attn(scores)
        attended = scores * weights
        return self.mlp(attended).squeeze(-1) * 100, weights
