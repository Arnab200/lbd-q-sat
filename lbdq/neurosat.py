import torch
import torch.nn as nn


def make_batch(problems):
    edge_lit = []
    edge_clause = []
    flip = []
    lit_graph = []
    labels = []
    lit_offset = 0
    clause_offset = 0

    for g, p in enumerate(problems):
        n = p["n_vars"]
        for c, clause in enumerate(p["clauses"]):
            for lit in clause:
                if lit > 0:
                    idx = lit - 1
                else:
                    idx = n - lit - 1
                edge_lit.append(lit_offset + idx)
                edge_clause.append(clause_offset + c)
        flip += [lit_offset + n + i for i in range(n)]
        flip += [lit_offset + i for i in range(n)]
        lit_graph += [g] * (2 * n)
        labels.append(p["label"])
        lit_offset += 2 * n
        clause_offset += len(p["clauses"])

    return {
        "edge_lit": torch.tensor(edge_lit),
        "edge_clause": torch.tensor(edge_clause),
        "flip": torch.tensor(flip),
        "lit_graph": torch.tensor(lit_graph),
        "labels": torch.tensor(labels, dtype=torch.float),
        "n_lits": lit_offset,
        "n_clauses": clause_offset,
        "n_graphs": len(problems),
    }


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def mlp(dim, out_dim):
    return nn.Sequential(
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Linear(dim, dim),
        nn.ReLU(),
        nn.Linear(dim, out_dim),
    )


class NeuroSAT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.L_init = nn.Parameter(torch.randn(1, dim) / dim ** 0.5)
        self.C_init = nn.Parameter(torch.randn(1, dim) / dim ** 0.5)
        self.L_msg = mlp(dim, dim)
        self.C_msg = mlp(dim, dim)
        self.L_update = nn.LSTMCell(2 * dim, dim)
        self.C_update = nn.LSTMCell(dim, dim)
        self.L_vote = mlp(dim, 1)

    def forward(self, batch, rounds, return_literals=False):
        n_lits = batch["n_lits"]
        n_clauses = batch["n_clauses"]
        edge_lit = batch["edge_lit"]
        edge_clause = batch["edge_clause"]
        flip = batch["flip"]
        device = self.L_init.device

        L_h = self.L_init.repeat(n_lits, 1)
        L_c = torch.zeros_like(L_h)
        C_h = self.C_init.repeat(n_clauses, 1)
        C_c = torch.zeros_like(C_h)

        for _ in range(rounds):
            messages = self.L_msg(L_h)[edge_lit]
            C_in = torch.zeros(n_clauses, self.dim, device=device).index_add(0, edge_clause, messages)
            C_h, C_c = self.C_update(C_in, (C_h, C_c))

            messages = self.C_msg(C_h)[edge_clause]
            L_in = torch.zeros(n_lits, self.dim, device=device).index_add(0, edge_lit, messages)
            L_h, L_c = self.L_update(torch.cat([L_in, L_h[flip]], dim=1), (L_h, L_c))

        votes = self.L_vote(L_h).squeeze(1)
        lit_graph = batch["lit_graph"]
        totals = torch.zeros(batch["n_graphs"], device=device).index_add(0, lit_graph, votes)
        counts = torch.zeros(batch["n_graphs"], device=device).index_add(0, lit_graph, torch.ones_like(votes))
        logits = totals / counts

        if return_literals:
            return logits, L_h
        return logits
