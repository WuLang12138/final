import numpy as np
import torch
import torch.nn as nn

from .layers import Attention
from .decoders import OmicsDecoder, CosineDecoder


def sparse_propagate(adj, x):
    return torch.spmm(adj, x)


class APPNPEncoder(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, propagation_steps=10, alpha=0.1):
        super().__init__()
        if propagation_steps < 0:
            raise ValueError("propagation_steps must be >= 0")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.propagation_steps = int(propagation_steps)
        self.alpha = float(alpha)
        self.lin1 = nn.Linear(nfeat, nhid)
        self.lin2 = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        h = torch.relu(self.lin1(x))
        h = self.dropout(h)
        h = self.lin2(h)
        h0 = h
        for _ in range(self.propagation_steps):
            h = (1.0 - self.alpha) * sparse_propagate(adj, h) + self.alpha * h0
        return h






class SiDMGF(nn.Module):
    def __init__(
        self,
        nfeat,
        nhid1,
        nhid2,
        dropout,
        encoder_type="appnp",
        spatial_encoder_type=None,
        signal_encoder_type=None,
        propagation_steps=10,
        propagation_alpha=0.1,
        gpr_init="ppr",
        gpr_softmax=False,
        signal_variant="dual",
        signal_propagation_steps=None,
        signal_propagation_alpha=None,
    ):
        super().__init__()
        encoder_type = str(encoder_type).lower()
        spatial_encoder_type = str(spatial_encoder_type or encoder_type).lower()
        signal_encoder_type = str(signal_encoder_type or encoder_type).lower()

        self.encoder_type = encoder_type
        self.spatial_encoder_type = spatial_encoder_type
        self.signal_encoder_type = signal_encoder_type
        if signal_variant not in {"dual", "spatial_only", "signal_identity"}:
            raise ValueError("Unknown signal_variant")
        self.signal_variant = signal_variant
        if signal_propagation_steps is not None and (signal_encoder_type != "appnp" or signal_propagation_steps < 0):
            raise ValueError("Separate signal steps require APPNP and a nonnegative value")
        if signal_propagation_alpha is not None and (signal_encoder_type != "appnp" or not 0 < signal_propagation_alpha <= 1):
            raise ValueError("Separate signal alpha requires APPNP and a value in (0, 1]")
        self.GNN_S = self._make_encoder(
            spatial_encoder_type,
            nfeat,
            nhid1,
            nhid2,
            dropout,
            propagation_steps,
            propagation_alpha,
            gpr_init,
            gpr_softmax,
        )
        self.GNN_F = self._make_encoder(
            signal_encoder_type,
            nfeat,
            nhid1,
            nhid2,
            dropout,
            propagation_steps,
            propagation_alpha,
            gpr_init,
            gpr_softmax,
        )
        self.ATT = Attention(nhid2)
        self.MLP = nn.Sequential(nn.Linear(nhid2, nhid2))
        self.omics_decoder = OmicsDecoder(nfeat, nhid1, nhid2)
        self.structure_decoder = CosineDecoder()
        self.dropout = nn.Dropout(p=dropout)
        # Initialize common modules exactly as in the paired full model first.
        if signal_propagation_steps is not None:
            self.GNN_F.propagation_steps = int(signal_propagation_steps)
        if signal_propagation_alpha is not None:
            self.GNN_F.alpha = float(signal_propagation_alpha)
        if signal_variant == "spatial_only":
            del self.GNN_F
            del self.ATT

    @staticmethod
    def _make_encoder(
        encoder_type,
        nfeat,
        nhid1,
        nhid2,
        dropout,
        propagation_steps,
        propagation_alpha,
        gpr_init,
        gpr_softmax,
    ):
        if encoder_type == "appnp":
            return APPNPEncoder(
                nfeat,
                nhid1,
                nhid2,
                dropout,
                propagation_steps=propagation_steps,
                alpha=propagation_alpha,
            )
        raise ValueError("Only APPNP is available in this base encoder")

    def forward(self, x, adj_s, adj_f):
        h_s = self.GNN_S(x, adj_s)
        if self.signal_variant == "spatial_only":
            h = h_s
            att = x.new_zeros((x.shape[0], 2, 1))
            att[:, 0] = 1
        else:
            if self.signal_variant == "signal_identity":
                indices = torch.arange(x.shape[0], device=x.device).repeat(2, 1)
                adj_f = torch.sparse_coo_tensor(indices, x.new_ones(x.shape[0]),
                                                adj_f.shape).coalesce()
            h_f = self.GNN_F(x, adj_f)
            h, att = self.ATT(torch.stack((h_s, h_f), dim=1))
        h = self.dropout(h)
        h = self.MLP(h)
        recon_adj = self.structure_decoder(h)
        pi, disp, mean = self.omics_decoder(h, adj_s)
        return h, recon_adj, pi, disp, mean, att


class NB(object):
    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(
            self.theta,
            torch.as_tensor(1e6, device=self.theta.device, dtype=self.theta.dtype),
        )
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(
            y_true + theta + self.eps
        )
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
            y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps))
        )
        final = _nan2inf(t1 + t2)
        if mean:
            final = torch.mean(final)
        return final


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(
            self.theta,
            torch.as_tensor(1e6, device=self.theta.device, dtype=self.theta.dtype),
        )
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        result += self.ridge_lambda * torch.square(self.pi)
        if mean:
            result = torch.mean(result)
        return _nan2inf(result)
