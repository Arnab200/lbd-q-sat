import random
import torch
import torch.nn as nn


def state_graph(solver):
    free = solver.unassigned()
    index = {v: i for i, v in enumerate(free)}
    edge_var = []
    edge_clause = []
    edge_sign = []
    n_clauses = 0

    for clause in solver.clauses[:solver.n_original]:
        if any(solver.value(lit) is True for lit in clause):
            continue
        free_lits = [lit for lit in clause if solver.value(lit) is None]
        if not free_lits:
            continue
        for lit in free_lits:
            edge_var.append(index[abs(lit)])
            edge_clause.append(n_clauses)
            edge_sign.append(1.0 if lit > 0 else -1.0)
        n_clauses += 1

    return {
        "vars": free,
        "edge_var": torch.tensor(edge_var, dtype=torch.long),
        "edge_clause": torch.tensor(edge_clause, dtype=torch.long),
        "edge_sign": torch.tensor(edge_sign, dtype=torch.float),
        "n_vars": len(free),
        "n_clauses": n_clauses,
    }


def batch_graphs(graphs):
    edge_var = []
    edge_clause = []
    edge_sign = []
    var_graph = []
    var_offset = 0
    clause_offset = 0

    for g, graph in enumerate(graphs):
        edge_var.append(graph["edge_var"] + var_offset)
        edge_clause.append(graph["edge_clause"] + clause_offset)
        edge_sign.append(graph["edge_sign"])
        var_graph += [g] * graph["n_vars"]
        var_offset += graph["n_vars"]
        clause_offset += graph["n_clauses"]

    return {
        "edge_var": torch.cat(edge_var),
        "edge_clause": torch.cat(edge_clause),
        "edge_sign": torch.cat(edge_sign),
        "var_graph": torch.tensor(var_graph, dtype=torch.long),
        "n_vars": var_offset,
        "n_clauses": clause_offset,
        "n_graphs": len(graphs),
    }


def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def mlp(in_dim, out_dim, hidden):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class QNetwork(nn.Module):
    def __init__(self, dim, rounds):
        super().__init__()
        self.dim = dim
        self.rounds = rounds
        self.var_init = nn.Parameter(torch.randn(1, dim) / dim ** 0.5)
        self.clause_init = nn.Parameter(torch.randn(1, dim) / dim ** 0.5)
        self.var_to_clause = mlp(dim + 1, dim, dim)
        self.clause_update = mlp(2 * dim, dim, dim)
        self.clause_to_var = mlp(dim + 1, dim, dim)
        self.var_update = mlp(2 * dim, dim, dim)
        self.clause_norm = nn.LayerNorm(dim)
        self.var_norm = nn.LayerNorm(dim)
        self.q_head = mlp(dim, 2, dim)

    def forward(self, batch):
        device = self.var_init.device
        edge_var = batch["edge_var"]
        edge_clause = batch["edge_clause"]
        sign = batch["edge_sign"].unsqueeze(1)

        var_h = self.var_init.repeat(batch["n_vars"], 1)
        clause_h = self.clause_init.repeat(batch["n_clauses"], 1)

        for _ in range(self.rounds):
            messages = self.var_to_clause(torch.cat([var_h[edge_var], sign], dim=1))
            clause_in = torch.zeros(batch["n_clauses"], self.dim, device=device).index_add(0, edge_clause, messages)
            clause_h = self.clause_norm(clause_h + self.clause_update(torch.cat([clause_h, clause_in], dim=1)))

            messages = self.clause_to_var(torch.cat([clause_h[edge_clause], sign], dim=1))
            var_in = torch.zeros(batch["n_vars"], self.dim, device=device).index_add(0, edge_var, messages)
            var_h = self.var_norm(var_h + self.var_update(torch.cat([var_h, var_in], dim=1)))

        return self.q_head(var_h)


def literal_from_action(graph, action):
    v = graph["vars"][action // 2]
    return v if action % 2 == 1 else -v


def choose_action(model, graph, epsilon, device):
    if random.random() < epsilon:
        return random.randrange(2 * graph["n_vars"])
    with torch.no_grad():
        q = model(to_device(batch_graphs([graph]), device))
    return q.view(-1).argmax().item()


def q_literal(model, device):
    def choose(solver):
        graph = state_graph(solver)
        return literal_from_action(graph, choose_action(model, graph, 0.0, device))
    return choose
