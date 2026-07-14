"""Shared SchNet model and molecular-feature utilities."""

from __future__ import annotations

import copy
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from torch.nn import BatchNorm1d, Dropout, Linear, PReLU
from torch_geometric.nn.models import SchNet
from torch_geometric.utils import scatter

SELECTED_DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumRotatableBonds",
    "FractionCSP3", "HeavyAtomCount", "NHOHCount", "NumAliphaticRings",
]
NODE_FEATURE_COLUMNS = [1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14, 15, 17, 18, 22]


def compute_global_descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(SELECTED_DESCRIPTOR_NAMES), dtype=np.float32)
    return np.asarray([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol), Descriptors.HeavyAtomCount(mol),
        Descriptors.NHOHCount(mol), Descriptors.NumAliphaticRings(mol),
    ], dtype=np.float32)


def prepare_graphs(data_list, original_indices, descriptor_matrix, scaler: StandardScaler):
    prepared = []
    for data, idx in zip(data_list, original_indices):
        graph = copy.deepcopy(data)
        graph.z = graph.x[:, 0].long()
        graph.node_attr = graph.x[:, NODE_FEATURE_COLUMNS].float()
        scaled = scaler.transform(descriptor_matrix[idx].reshape(1, -1)).astype(np.float32)
        graph.global_desc = torch.tensor(scaled, dtype=torch.float)
        prepared.append(graph)
    return prepared


class SchNetEmbeddingGlobalDescRegressor(torch.nn.Module):
    def __init__(self, num_global_features: int, num_node_features: int,
                 hidden_channels: int = 64, num_filters: int = 64,
                 num_interactions: int = 4, num_gaussians: int = 50,
                 cutoff: float = 8.0, dropout: float = 0.10) -> None:
        super().__init__()
        self.schnet = SchNet(
            hidden_channels=hidden_channels, num_filters=num_filters,
            num_interactions=num_interactions, num_gaussians=num_gaussians,
            cutoff=cutoff, readout="mean",
        )
        self.node_encoder = Linear(num_node_features, hidden_channels)
        self.batch_norm = BatchNorm1d(hidden_channels + num_global_features)
        self.linear_1 = Linear(hidden_channels + num_global_features, 128)
        self.prelu_1 = PReLU()
        self.dropout = Dropout(dropout)
        self.linear_2 = Linear(128, 64)
        self.prelu_2 = PReLU()
        self.output = Linear(64, 1)

    def encode_3d(self, z, pos, batch, node_attr):
        hidden = self.schnet.embedding(z) + self.node_encoder(node_attr)
        edge_index, edge_weight = self.schnet.interaction_graph(pos, batch)
        edge_attr = self.schnet.distance_expansion(edge_weight)
        for interaction in self.schnet.interactions:
            hidden = hidden + interaction(hidden, edge_index, edge_weight, edge_attr)
        return scatter(hidden, batch, dim=0, dim_size=int(batch.max()) + 1, reduce="mean")

    def forward(self, z, pos, batch, global_desc, node_attr):
        hidden = self.encode_3d(z, pos, batch, node_attr)
        global_desc = global_desc.view(hidden.size(0), -1)
        features = torch.cat([hidden, global_desc], dim=1)
        features = self.batch_norm(features)
        features = self.prelu_1(self.linear_1(features))
        features = self.dropout(features)
        features = self.prelu_2(self.linear_2(features))
        return self.output(features).view(-1)
