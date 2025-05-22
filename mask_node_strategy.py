import torch



def random_mask_node(input_features, num_masked_nodes):
    batch_size, num_nodes, num_features = input_features.size()
    indices = torch.randperm(num_nodes)[:num_masked_nodes]
    masked_features = input_features.clone()
    masked_features[:, indices, :] = 0  # Set masked nodes to 0
    return masked_features



def random_mask_node_mode(input_features, masked_nodes, n, mode=1):
    batch_size, num_nodes, num_features = input_features.size()
    masked_features = input_features.clone()

    if mode == 1:
        # 在指定的节点中随机选择n个节点
        random_indices = torch.randperm(len(masked_nodes))[:n]
        selected_nodes = [masked_nodes[i] for i in random_indices]
    elif mode == 2:
        # 在除了指定的节点以外的所有节点中随机选择n个节点
        all_nodes = set(range(num_nodes))
        unmasked_nodes = list(all_nodes - set(masked_nodes))
        random_indices = torch.randperm(len(unmasked_nodes))[:n]
        selected_nodes = [unmasked_nodes[i] for i in random_indices]
    elif mode == 3:
        # 在所有节点中随机选择n个节点
        all_nodes = list(range(num_nodes))
        random_indices = torch.randperm(num_nodes)[:n]
        selected_nodes = [all_nodes[i] for i in random_indices]
    else:
        raise ValueError("Invalid mode. Mode must be 1, 2, or 3.")

    # 将选定的节点置为0
    masked_features[:, selected_nodes, :] = 0

    return masked_features





if __name__ == '__main__':

    # 创建一个示例特征张量
    batch_size = 2
    num_nodes = 54
    num_features = 10
    input_features = torch.randn(batch_size, num_nodes, num_features)

    # 指定需要掩码的节点索引列表
    masked_nodes = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]

    # 对指定节点进行掩码
    masked_features = random_mask_node_mode(input_features, masked_nodes, n=10, mode=1)

    # 输出结果
    print("原始特征张量大小：", input_features.size())
    print("掩码后特征张量大小：", masked_features.size())



