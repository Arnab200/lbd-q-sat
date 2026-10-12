"""
models.py

Model definitions for GNN-based MINST Classifier
Import GNN (or GCN/GAT variants) from here into your own train.py/
test.py/ validate.py scripts

Example:
    from models import GNN, build_grid_edge_index()

    edge_index = build_grid_edge_index()
    model = GNN(in_channels=1, hidden_channels=32, num_classes=10)

"""

import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

def build_grid_edge_index(height=28, width=28):
    """
    Builds the fixed 8 neightbour grid adjacency for a height x width image.
    Pixel(r, c) is flattened to index r * width + c. Reused for every image since MINST images
    all share the same 28x28 grid topology.
    """

    edges = []
    for r in range(height):
        for c in range(width):
            idx = r * width + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        nidx = 