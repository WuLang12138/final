"""Isolated encoder adapters; no changes to historical model implementations."""
import torch
from torch import nn
from .propagation import SiDMGF




class ACMStyleLayer(nn.Module):
    """ACM-inspired adaptation, NOT an exact official ACM reproduction.

    Independent channel transforms: P X W_low, (I-P) X W_high, X W_identity.
    Node-wise learned softmax gates, no labels or added loss. No factor-of-three
    output scaling or official sigmoid/temperature attention mixer.
    """
    def __init__(self, nin, nout):
        super().__init__()
        self.transforms = nn.ModuleList([nn.Linear(nin, nout, bias=False) for _ in range(3)])
        self.gates = nn.ModuleList([nn.Linear(nout, 1) for _ in range(3)])
        self.bias = nn.Parameter(torch.zeros(nout))

    def forward(self, x, adj):
        low, high, identity = [layer(x) for layer in self.transforms]
        channels = [torch.spmm(adj, low), high - torch.spmm(adj, high), identity]
        weights = torch.softmax(torch.cat([g(h) for g, h in zip(self.gates, channels)], -1), -1)
        return sum(h * weights[:, k:k+1] for k, h in enumerate(channels)) + self.bias


class TwoLayerEncoder(nn.Module):
    def __init__(self, layer, nin, hidden, out, dropout):
        super().__init__()
        self.first, self.second = layer(nin, hidden), layer(hidden, out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        return self.second(self.dropout(torch.relu(self.first(x, adj))), adj)


class ComparisonSiDMGF(SiDMGF):
    @staticmethod
    def _make_encoder(kind, nin, hidden, out, dropout, steps, alpha, init, softmax):
        if kind == 'acm_style':
            layer = ACMStyleLayer
            return TwoLayerEncoder(layer, nin, hidden, out, dropout)
        return SiDMGF._make_encoder(kind, nin, hidden, out, dropout, steps, alpha, init, softmax)
