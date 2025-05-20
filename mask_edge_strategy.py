import torch

import torch

def random_mask_edge_mode(adj_matrix, masked_nodes, mode=1, ratio=1.0):
    num_nodes, _ = adj_matrix.size()
    masked_adj_matrix = adj_matrix.clone()

    if mode == 1:
        # Randomly remove a certain percentage of edges, which only connect to the specified 24 nodes
        num_edges_to_mask = int(ratio * num_nodes * (num_nodes - 1) / 2)
        for _ in range(num_edges_to_mask):
            # Randomly select an edge between the 24 specified nodes and remove it
            node_idx = torch.randint(0, len(masked_nodes), (1,))
            node1 = masked_nodes[node_idx]
            node2 = masked_nodes[(node_idx + torch.randint(1, len(masked_nodes), (1,))) % len(masked_nodes)]
            masked_adj_matrix[node1, node2] = 0
            masked_adj_matrix[node2, node1] = 0
    elif mode == 2:
        # Remove edges that are not connected to the specified 24 nodes
        other_nodes = set(range(num_nodes)) - set(masked_nodes)
        num_other_nodes = len(other_nodes)
        num_edges_to_mask = int(ratio * num_other_nodes * (num_other_nodes - 1) / 2)
        for _ in range(num_edges_to_mask):
            # Randomly select an edge between nodes outside the 24 specified nodes and remove it
            node1 = torch.randint(0, num_nodes, (1,))
            node2 = torch.randint(0, num_nodes, (1,))
            while node1 in masked_nodes or node2 in masked_nodes or node1 == node2:
                node1 = torch.randint(0, num_nodes, (1,))
                node2 = torch.randint(0, num_nodes, (1,))
            masked_adj_matrix[node1, node2] = 0
            masked_adj_matrix[node2, node1] = 0
    elif mode == 3:
        # Randomly remove edges connecting the specified 24 nodes and the other 54-24 nodes
        other_nodes = set(range(num_nodes)) - set(masked_nodes)
        num_other_nodes = len(other_nodes)
        num_edges_to_mask = int(ratio * len(masked_nodes) * num_other_nodes)
        for _ in range(num_edges_to_mask):
            # Randomly select an edge between the 24 specified nodes and the other 54-24 nodes and remove it
            node1_idx = torch.randint(0, len(masked_nodes), (1,))
            node1 = masked_nodes[node1_idx]
            node2 = torch.randint(0, num_nodes, (1,))
            while node2 in masked_nodes or node1 == node2:
                node2 = torch.randint(0, num_nodes, (1,))
            masked_adj_matrix[node1, node2] = 0
            masked_adj_matrix[node2, node1] = 0
    else:
        raise ValueError("Invalid mode. Mode must be 1, 2, or 3.")

    return masked_adj_matrix



def random_mask_edge_mode_batch(adj_matrix, masked_nodes, mode=1, ratio=0.1):
    batch_size, num_nodes, _ = adj_matrix.size()
    masked_adj_matrix = adj_matrix.clone()

    if mode == 1:
        # 随机去掉一定比例的边，这些边只连接24个指定的节点
        num_edges_to_mask = int(ratio * num_nodes * (num_nodes - 1) / 2)
        for _ in range(num_edges_to_mask):
            # 在24个指定的节点之间随机选择一条边并去掉
            node_idx = torch.randint(0, len(masked_nodes), (1,))
            node1 = masked_nodes[node_idx]
            node2 = masked_nodes[(node_idx + torch.randint(1, len(masked_nodes), (1,))) % len(masked_nodes)]
            masked_adj_matrix[:, node1, node2] = 0
            masked_adj_matrix[:, node2, node1] = 0
    elif mode == 2:
        # 去掉的边是在除了24个指定的节点之外的
        other_nodes = set(range(num_nodes)) - set(masked_nodes)
        num_other_nodes = len(other_nodes)
        num_edges_to_mask = int(ratio * num_other_nodes * (num_other_nodes - 1) / 2)
        for _ in range(num_edges_to_mask):
            # 在除了24个指定的节点以外的节点之间随机选择一条边并去掉
            node1 = torch.randint(0, num_nodes, (1,))
            node2 = torch.randint(0, num_nodes, (1,))
            while node1 in masked_nodes or node2 in masked_nodes or node1 == node2:
                node1 = torch.randint(0, num_nodes, (1,))
                node2 = torch.randint(0, num_nodes, (1,))
            masked_adj_matrix[:, node1, node2] = 0
            masked_adj_matrix[:, node2, node1] = 0
    elif mode == 3:
        # 随机去掉的边是连接24个指定的节点和其他54-24个节点的
        other_nodes = set(range(num_nodes)) - set(masked_nodes)
        num_other_nodes = len(other_nodes)
        num_edges_to_mask = int(ratio * len(masked_nodes) * num_other_nodes)
        for _ in range(num_edges_to_mask):
            # 在24个指定的节点和其他54-24个节点之间随机选择一条边并去掉
            node1_idx = torch.randint(0, len(masked_nodes), (1,))
            node1 = masked_nodes[node1_idx]
            node2 = torch.randint(0, num_nodes, (1,))
            while node2 in masked_nodes or node1 == node2:
                node2 = torch.randint(0, num_nodes, (1,))
            masked_adj_matrix[:, node1, node2] = 0
            masked_adj_matrix[:, node2, node1] = 0
    else:
        raise ValueError("Invalid mode. Mode must be 1, 2, or 3.")

    return masked_adj_matrix


if __name__ == '__main__':
    import torch

    # 示例输入：邻接矩阵大小为 8x54x54
    adj_matrix = torch.randint(0, 2, size=(54, 54))

    # 示例被屏蔽节点列表
    masked_nodes = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]


    # 示例调用函数
    masked_adj_matrix = random_mask_edge_mode(adj_matrix, masked_nodes, mode=1, ratio=0.2)

    print("原始邻接矩阵：")
    print(adj_matrix.shape)
    print("\n被屏蔽边后的邻接矩阵：")
    print(masked_adj_matrix.shape)
