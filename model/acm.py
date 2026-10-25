"""Experimental, label-free framework adaptations, not official ACM reproductions."""
import torch
from torch import nn
from .encoder_adapters import ComparisonSiDMGF
from .decoders import OmicsDecoder

ARMS = {'acm_split_noaux': ('acm_style', 0.0)}


class LocalOmicsDecoder(OmicsDecoder):
    """Exactly the existing decoder, but mean uses identity adjacency."""
    def forward(self, emb, adj):
        ids = torch.arange(len(emb), device=emb.device).repeat(2, 1)
        identity = torch.sparse_coo_tensor(ids, emb.new_ones(len(emb)), (len(emb),len(emb))).coalesce()
        return super().forward(emb, identity)


class SplitFrequencyEncoder(nn.Module):
    """ACM-inspired common/detail separation; 32+32 dimensions, signed detail.

    This is a new experimental architecture, NOT the earlier ACM-style encoder.
    Structure reconstruction acts on common only; expression uses both channels.
    """
    def __init__(self, nin, hidden, out, dropout):
        super().__init__()
        if out % 2: raise ValueError('Split latent dimension must be even')
        self.pre = nn.Sequential(nn.Linear(nin, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.low = nn.Linear(hidden, out//2)
        self.high = nn.Linear(hidden, out//2)
        self.identity = nn.Linear(hidden, out//2)
        self.low_norm, self.detail_norm = nn.LayerNorm(out//2), nn.LayerNorm(out//2)
        self.gate = nn.Linear(out, out//2)

    def forward(self, x, adj):
        h = self.pre(x)
        ph = torch.spmm(adj, h)
        self.common = self.low_norm(self.low(ph))
        self.detail = self.detail_norm(self.high(h-ph) + self.identity(h))
        self.last_gate = torch.sigmoid(self.gate(torch.cat([self.common, self.detail], 1)))
        return torch.cat([self.common, self.last_gate*self.detail], 1)

class AdaptedSiDMGF(ComparisonSiDMGF):
    def __init__(self, *args, framework, **kwargs):
        super().__init__(*args, **kwargs)
        self.framework = framework
        self.aux_weight = ARMS[framework][1]
        nin, hidden, out, drop = [kwargs[k] for k in ('nfeat','nhid1','nhid2','dropout')]
        # Preserve decoder parameter initialization when removing graph mixing.
        self.omics_decoder.__class__ = LocalOmicsDecoder
        self.split = framework.startswith('acm_split')
        if self.split:
            self.GNN_S = SplitFrequencyEncoder(nin, hidden, out, drop)
            self.common_head = nn.Linear(out//2, 32)
            self.detail_head = nn.Linear(out//2, 32)
        self.register_buffer('target_low', None, persistent=False)
        self.register_buffer('target_high', None, persistent=False)

    def prepare_targets(self, x, adj):
        if not self.split or self.target_low is not None: return
        # Frozen, label-free PCA reference, same random_state for all seeds.
        import numpy as np
        from sklearn.decomposition import PCA
        a = np.log1p(x.detach().cpu().numpy())
        a = (a-a.mean(0)) / np.maximum(a.std(0), 1e-4)
        d = min(32, a.shape[0]-1, a.shape[1])
        z = PCA(n_components=d, svd_solver='randomized', random_state=0).fit_transform(a)
        z = torch.as_tensor(z, device=x.device, dtype=x.dtype)
        if d < 32: z = torch.nn.functional.pad(z, (0,32-d))
        low = torch.spmm(adj,z)
        high = z-low
        self.target_low = low / low.square().mean().sqrt().clamp_min(1e-6)
        self.target_high = high / high.square().mean().sqrt().clamp_min(1e-6)

    def forward(self, x, adj_s, adj_f):
        encoded = self.GNN_S(x, adj_s)
        h = self.MLP(self.dropout(encoded))
        structure_latent = self.GNN_S.common if self.split else h
        rec_adj = self.structure_decoder(structure_latent)
        pi, disp, mean = self.omics_decoder(h, adj_s)
        att = x.new_zeros((len(x),2,1)); att[:,0] = 1
        self.aux_low = x.new_zeros(())
        self.aux_high = x.new_zeros(())
        if self.split:
            self.prepare_targets(x, adj_s)
            self.aux_low = torch.nn.functional.mse_loss(self.common_head(self.GNN_S.common), self.target_low)
            # Train the gated detail actually supplied to the final embedding.
            self.aux_high = torch.nn.functional.mse_loss(self.detail_head(self.GNN_S.last_gate*self.GNN_S.detail), self.target_high)
        return h, rec_adj, pi, disp, mean, att
