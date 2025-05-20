import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_scatter import scatter_max


def adj_to_edge_index_attr(adj_matrix):
    # adj_matrix's shape: [Batch, time_series, roi_number]
    edge_index_list = []
    edge_attr_list = []
    num_nodes = adj_matrix.shape[1]
    # For each batch
    for i in range(adj_matrix.shape[0]):
        # Get the adjacency matrix for the current batch
        adj = adj_matrix[i]
        # Get the indices and values of the non-zero elements of the adjacency matrix
        nonzero = torch.nonzero(adj)
        values = adj[nonzero[:, 0], nonzero[:, 1]]

        indices = torch.cat([nonzero[:, 1:], nonzero[:, :1].repeat(1, 2)], dim=1)
        indices[:, 0] += i * num_nodes
        indices[:, 1] += i * num_nodes

        edge_index_list.append(indices)
        edge_attr_list.append(values)
    edge_index = torch.cat(edge_index_list, dim=0).t().contiguous()
    edge_attr = torch.cat(edge_attr_list, dim=0)
    edge_index[0] -= num_nodes
    edge_index = edge_index[1:]
    return edge_index, edge_attr

class GCN_all(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels):
        super(GCN_all, self).__init__()
        self.conv1 = GCNConv(in_channels, hid_channels)
        self.conv2 = GCNConv(hid_channels, out_channels)
        self.linear1 = torch.nn.Linear(64, 32)
        self.linear2 = torch.nn.Linear(32, 2)

        self.Bn1 = nn.BatchNorm1d(in_channels)

    def forward(self,
                time_seires: torch.tensor,
                node_features: torch.tensor):
        num_nodes = node_features.shape[-1]
        batch_size = node_features.shape[0]
        x = time_seires.view(-1, num_nodes)
        edge_index, edge_attr = adj_to_edge_index_attr(node_features)
        # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        batch = torch.arange(batch_size).repeat_interleave(x.shape[0] // batch_size).cuda()
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)

        x, _ = scatter_max(x, batch, dim=0)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x