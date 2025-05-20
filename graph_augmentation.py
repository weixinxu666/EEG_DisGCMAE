import torch
import torch.nn.functional as F
import random
import torch.nn as nn


def drop_nodes(node_features, drop_prob, drop_ratio):
    """
    Drop a certain ratio of nodes in the graph.

    Args:
    - node_features (torch.Tensor): Node features tensor of shape (batch_size, num_nodes, feature_dim).
    - drop_prob (float): Probability of dropping a node.
    - drop_ratio (float): Ratio of nodes to be dropped.

    Returns:
    - torch.Tensor: Node features tensor with dropped nodes.
    """
    batch_size, num_nodes, feature_dim = node_features.size()
    mask = torch.rand(batch_size, num_nodes) > drop_prob
    for i in range(batch_size):
        drop_num = int(num_nodes * drop_ratio)
        drop_indices = torch.randperm(num_nodes)[:drop_num]
        mask[i, drop_indices] = False
    mask = mask.unsqueeze(-1).expand(-1, -1, feature_dim)
    mask = mask.cuda()
    return node_features * mask


def remove_edges(adjacency_matrix, remove_prob, remove_ratio):
    """
    Remove a certain ratio of edges in the graph.

    Args:
    - adjacency_matrix (torch.Tensor): Adjacency matrix of shape (batch_size, num_nodes, num_nodes).
    - remove_prob (float): Probability of removing an edge.
    - remove_ratio (float): Ratio of edges to be removed.

    Returns:
    - torch.Tensor: Adjacency matrix with removed edges.
    """
    batch_size, num_nodes, _ = adjacency_matrix.size()
    mask = torch.rand(batch_size, num_nodes, num_nodes) > remove_prob
    for i in range(batch_size):
        edge_indices = torch.nonzero(adjacency_matrix[i], as_tuple=False)
        num_edges = edge_indices.size(0)
        remove_num = int(num_edges * remove_ratio)
        remove_indices = edge_indices[torch.randperm(num_edges)[:remove_num]]
        mask[i, remove_indices[:, 0], remove_indices[:, 1]] = False
        mask = mask.cuda()
    return adjacency_matrix * mask


def add_noise_to_features(node_features, noise_level):
    """
    Add noise to node features.

    Args:
    - node_features (torch.Tensor): Node features tensor of shape (batch_size, num_nodes, feature_dim).
    - noise_level (float): Standard deviation of the Gaussian noise to add.

    Returns:
    - torch.Tensor: Node features tensor with added noise.
    """
    noise = torch.randn_like(node_features) * noise_level
    return node_features + noise


def graph_aug(adjacency_matrix, node_features, drop_prob=0.2, drop_ratio=0.2, remove_prob=0.2,
                       remove_ratio=0.2, noise_level=0.1):
    """
    Perform graph augmentation including node dropping, edge removing, and adding noise to node features.

    Args:
    - adjacency_matrix (torch.Tensor): Adjacency matrix of shape (batch_size, num_nodes, num_nodes).
    - node_features (torch.Tensor): Node features tensor of shape (batch_size, num_nodes, feature_dim).
    - drop_prob (float): Probability of dropping a node.
    - drop_ratio (float): Ratio of nodes to be dropped.
    - remove_prob (float): Probability of removing an edge.
    - remove_ratio (float): Ratio of edges to be removed.
    - noise_level (float): Standard deviation of the Gaussian noise to add to node features.

    Returns:
    - torch.Tensor: Augmented adjacency matrix.
    - torch.Tensor: Augmented node features.
    """
    augmented_node_features = drop_nodes(node_features, drop_prob, drop_ratio)
    augmented_node_features = add_noise_to_features(augmented_node_features, noise_level)
    augmented_adjacency_matrix = remove_edges(adjacency_matrix, remove_prob, remove_ratio)

    return augmented_adjacency_matrix, augmented_node_features



def graph_aug_feat(node_features, drop_prob=0.2, drop_ratio=0.2, remove_prob=0.2,
                       remove_ratio=0.2, noise_level=0.1):
    """
    Perform graph augmentation including node dropping, edge removing, and adding noise to node features.

    Args:
    - adjacency_matrix (torch.Tensor): Adjacency matrix of shape (batch_size, num_nodes, num_nodes).
    - node_features (torch.Tensor): Node features tensor of shape (batch_size, num_nodes, feature_dim).
    - drop_prob (float): Probability of dropping a node.
    - drop_ratio (float): Ratio of nodes to be dropped.
    - remove_prob (float): Probability of removing an edge.
    - remove_ratio (float): Ratio of edges to be removed.
    - noise_level (float): Standard deviation of the Gaussian noise to add to node features.

    Returns:
    - torch.Tensor: Augmented adjacency matrix.
    - torch.Tensor: Augmented node features.
    """
    augmented_node_features = drop_nodes(node_features, drop_prob, drop_ratio)
    augmented_node_features = add_noise_to_features(augmented_node_features, noise_level)

    return augmented_node_features


import torch


def graph_aug_feat_moco(node_features, drop_ratio=0.6, noise_level=0.2):
    # 添加噪声并移除节点（对 query 和 key）
    query = remove_nodes_and_add_noise(node_features.clone(), drop_ratio, noise_level, seed=88)
    key = remove_nodes_and_add_noise(node_features.clone(), drop_ratio, noise_level, seed=32)

    # 生成随机索引
    num_nodes = node_features.size(1)
    indices_query = torch.randint(0, num_nodes, (int(drop_ratio * num_nodes),), device=node_features.device)
    indices_key = torch.randint(0, num_nodes, (int(drop_ratio * num_nodes),), device=node_features.device)

    # 将移除的节点用嵌入向量替换（对 query）
    query_emb = replace_nodes_with_embedding(query.clone(), indices_query)

    # 将移除的节点用嵌入向量替换（对 key）
    key_emb = replace_nodes_with_embedding(key.clone(), indices_key)

    # 在移除节点和添加噪声后检查节点是否为零
    # check_zero_nodes(query, indices_query)
    # check_zero_nodes(key, indices_key)

    return query, key, query_emb, key_emb


def remove_nodes_and_add_noise(node_features, drop_ratio, noise_level, seed=None):
    """
    移除节点并添加噪声
    """
    if seed is not None:
        torch.manual_seed(seed)

    num_nodes = node_features.size(1)
    num_remove = int(drop_ratio * num_nodes)

    for i in range(node_features.size(0)):
        indices_remove = torch.randperm(num_nodes)[:num_remove]
        node_features[i, indices_remove, :] = 0  # 移除节点，将其特征置零

    noise = torch.randn_like(node_features) * noise_level
    return node_features + noise


def replace_nodes_with_embedding(node_features, indices):
    """
    将指定索引的节点特征替换为可学习的嵌入向量
    """
    # 创建一个可学习的嵌入层，并将其移动到与输入张量相同的设备上
    embedding_layer = torch.nn.Embedding(len(indices), node_features.size(-1)).to(node_features.device)
    # 通过嵌入层获取嵌入向量
    embedded_features = embedding_layer(torch.arange(len(indices), device=node_features.device))
    # 替换被移除节点的特征为嵌入向量
    node_features[:, indices, :] = embedded_features.unsqueeze(0)
    return node_features


def check_zero_nodes(node_features, indices):
    for i in range(node_features.size(0)):
        zero_indices = torch.where(node_features[i, indices, :].sum(dim=1) == 0)[0]
        if len(zero_indices) > 0:
            print(f"Warning: Some nodes at indices {zero_indices} in batch {i} are not zero.")
        else:
            print(f"All nodes at indices {indices} in batch {i} are set to zero.")




if __name__ == '__main__':
    # Example usage
    batch_size = 32
    num_nodes = 54
    feature_dim = 128

    adjacency_matrix = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()
    node_features = torch.randn(batch_size, num_nodes, feature_dim)

    node_features = node_features.cuda()

    q, k, q_e, k_e = graph_aug_feat_moco(node_features, drop_ratio=0.2,  noise_level=0.1)

    print("done")

    # print(q)



