"""GNN for 86-way DDI type prediction.

Encoder: AutoDDI's searched DrugBank-inductive architecture (logger/drugbank_inductive_search
in github.com/Zhen-Peng-Wu/AutoDDI): 8 cells, each a GCNConv over the molecule's own
bonds plus a GATConv over the full atom-atom bipartite graph to its partner, outputs
concatenated (hidden/2 each), then graph LayerNorm and ELU. Weights are shared between
the two drugs.

Readout: AutoDDI's searched readout is SAGPool + global mean. Here it's replaced by
co-attention pooling (MGDDI's finding is that partner-conditioned pooling does the
most work): at every layer, each drug's atoms are scored against the partner's mean
atom embedding and pooled with those weights, so the same drug gets a different
vector in each pair. readout="mean" swaps in plain global mean pooling for the ablation.

Classifier: per-layer pooled vectors of both drugs, concatenated in (Drug1, Drug2)
order since many DDI types are directional, then an MLP to 86 logits.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, LayerNorm, global_mean_pool
from torch_geometric.utils import softmax

from graphs import ATOM_DIM


def bipartite_edges(ptr_a: torch.Tensor, ptr_b: torch.Tensor) -> torch.Tensor:
    """All (atom of B, atom of A) edges within each pair, for B -> A message passing.

    ptr_a/ptr_b are the PyG Batch.ptr vectors of the two drug batches, where graph k
    of each batch belongs to pair k.
    """
    n_a = ptr_a[1:] - ptr_a[:-1]
    n_b = ptr_b[1:] - ptr_b[:-1]
    n_edges = n_a * n_b
    device = ptr_a.device
    pair = torch.repeat_interleave(torch.arange(len(n_a), device=device), n_edges)
    k = torch.arange(int(n_edges.sum()), device=device) - torch.repeat_interleave(
        torch.cumsum(n_edges, 0) - n_edges, n_edges
    )
    src = ptr_b[pair] + k % n_b[pair]
    dst = ptr_a[pair] + k // n_b[pair]
    return torch.stack([src, dst])


class Cell(nn.Module):
    def __init__(self, in_dim: int, hidden: int, heads: int, inter: bool):
        super().__init__()
        self.inter = inter
        intra_dim = hidden // 2 if inter else hidden
        self.conv = GCNConv(in_dim, intra_dim)
        if inter:
            self.bi_conv = GATConv((in_dim, in_dim), hidden // 2 // heads, heads=heads)

    def forward(self, x_a, x_b, edge_a, edge_b, bi_ba):
        x_a, x_b = F.elu(x_a), F.elu(x_b)
        h_a, h_b = self.conv(x_a, edge_a), self.conv(x_b, edge_b)
        if not self.inter:
            return h_a, h_b
        bi_a = self.bi_conv((x_b, x_a), bi_ba)
        bi_b = self.bi_conv((x_a, x_b), bi_ba.flip(0))
        return torch.cat([h_a, bi_a], 1), torch.cat([h_b, bi_b], 1)


class CoAttentionPool(nn.Module):
    """Pool one drug's atoms with weights conditioned on its partner."""

    def __init__(self, hidden: int):
        super().__init__()
        self.w_atom = nn.Linear(hidden, hidden, bias=False)
        self.w_partner = nn.Linear(hidden, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, x, batch, partner_summary):
        e = self.score(torch.tanh(self.w_atom(x) + self.w_partner(partner_summary)[batch]))
        alpha = softmax(e.squeeze(-1), batch)
        pooled = x.new_zeros(partner_summary.shape).index_add_(
            0, batch, alpha.unsqueeze(-1) * x
        )
        return pooled, alpha


class DDIModel(nn.Module):
    def __init__(
        self,
        n_classes: int = 86,
        n_layers: int = 8,
        hidden: int = 128,
        heads: int = 2,
        inter: bool = True,
        readout: str = "coattn",
        dropout: float = 0.2,
        gnn_dropout: float = 0.0,
    ):
        super().__init__()
        assert readout in {"coattn", "mean"}
        self.readout = readout
        self.inter = inter
        self.gnn_dropout = gnn_dropout
        self.input_norm = LayerNorm(ATOM_DIM)
        self.cells = nn.ModuleList(
            Cell(ATOM_DIM if i == 0 else hidden, hidden, heads, inter) for i in range(n_layers)
        )
        self.norms = nn.ModuleList(LayerNorm(hidden) for _ in range(n_layers))
        if readout == "coattn":
            self.pools = nn.ModuleList(CoAttentionPool(hidden) for _ in range(n_layers))
        self.classifier = nn.Sequential(
            nn.Linear(2 * n_layers * hidden, 512),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_classes),
        )

    def forward(self, a, b, return_attention: bool = False):
        """a, b: PyG Batches of Drug1 and Drug2 graphs, graph k of each = pair k."""
        x_a, x_b = self.input_norm(a.x, a.batch), self.input_norm(b.x, b.batch)
        bi_ba = bipartite_edges(a.ptr, b.ptr) if self.inter else None

        reps_a, reps_b, attn_a, attn_b = [], [], [], []
        for cell, norm, i in zip(self.cells, self.norms, range(len(self.cells))):
            x_a, x_b = cell(x_a, x_b, a.edge_index, b.edge_index, bi_ba)
            x_a, x_b = F.elu(norm(x_a, a.batch)), F.elu(norm(x_b, b.batch))
            x_a = F.dropout(x_a, self.gnn_dropout, self.training)
            x_b = F.dropout(x_b, self.gnn_dropout, self.training)

            mean_a = global_mean_pool(x_a, a.batch, a.num_graphs)
            mean_b = global_mean_pool(x_b, b.batch, b.num_graphs)
            if self.readout == "mean":
                reps_a.append(mean_a)
                reps_b.append(mean_b)
            else:
                r_a, al_a = self.pools[i](x_a, a.batch, mean_b)
                r_b, al_b = self.pools[i](x_b, b.batch, mean_a)
                reps_a.append(r_a)
                reps_b.append(r_b)
                attn_a.append(al_a)
                attn_b.append(al_b)

        logits = self.classifier(torch.cat(reps_a + reps_b, dim=1))
        if return_attention:
            return logits, attn_a, attn_b
        return logits
