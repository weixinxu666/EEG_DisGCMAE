import torch
import torch.nn.functional as F

def Eu_dis(x):
    """
    Calculate the distance among each row of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    aa = torch.sum(x * x, dim=1).unsqueeze(1)
    ab = torch.matmul(x, x.t())
    dist_mat = aa + aa.t() - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = torch.sqrt(dist_mat)
    dist_mat = torch.max(dist_mat, dist_mat.t())
    return dist_mat


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypergraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.size(0)
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = torch.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = torch.argsort(dis_vec)
        avg_dis = torch.mean(dis_vec)
        if center_idx not in nearest_idx[:k_neig]:
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = torch.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs, split_diff_scale=False, is_probH=True, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.view(-1, X.shape[-1])

    if isinstance(K_neigs, int):
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H


def adj2H(x, k_neigs):
    H = construct_H_with_KNN(x, K_neigs=[k_neigs], split_diff_scale=True)

    # 根据关联矩阵生成超图的邻接矩阵
    G = generate_G_from_H(H)

    return G[0]


def adj2H_aug(x, k_neigs):
    H = construct_H_with_KNN(x, K_neigs=[k_neigs], split_diff_scale=True)

    # 根据关联矩阵生成超图的邻接矩阵
    G = generate_G_from_H(H)

    return G[0]


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if isinstance(h, list):
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(torch.cat((a, b), dim=1))
                    H = tmp
                else:
                    H = torch.cat((H, h), dim=1)
    return H


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if isinstance(H, list):
        G = []
        for sub_H in H:
            G.append(_generate_G_from_H(sub_H, variable_weight))
        return G
    else:
        return _generate_G_from_H(H, variable_weight)


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    n_edge = H.size(1)
    # the weight of the hyperedge
    W = torch.ones(n_edge)
    # the degree of the node
    DV = torch.sum(H * W, dim=1)
    # the degree of the hyperedge
    DE = torch.sum(H, dim=0)

    invDE = torch.diag(torch.pow(DE, -1))
    DV2 = torch.diag(torch.pow(DV, -0.5))
    W = torch.diag(W)

    if variable_weight:
        DV2_H = torch.matmul(DV2, H)
        invDE_HT_DV2 = torch.matmul(torch.matmul(invDE, H.t()), DV2)
        return DV2_H, W, invDE_HT_DV2
    else:
        G = torch.matmul(DV2, torch.matmul(H, torch.matmul(W, torch.matmul(invDE, torch.matmul(H.t(), DV2)))))
        return G


def random_mask_H_with_ratio(H, ratio, mode):
    """
    Randomly mask elements of the incidence matrix H based on the specified mode and ratio.

    Parameters:
        H (torch.Tensor): Incidence matrix of the hypergraph.
        ratio (float): Percentage of elements to mask, ranges from 0 to 1.
        mode (int): Mode of masking (1, 2, or 3).

    Returns:
        torch.Tensor: Masked incidence matrix.
    """
    masked_H = H.clone()

    if mode == 1:  # Randomly select rows and set their non-zero elements to zero
        rows_to_mask = int(H.size(0) * ratio)
        rows = torch.randperm(H.size(0))[:rows_to_mask]
        for row in rows:
            masked_H[row, H[row, :] != 0] = 0

    elif mode == 2:  # Randomly select columns and set their non-zero elements to zero
        cols_to_mask = int(H.size(1) * ratio)
        cols = torch.randperm(H.size(1))[:cols_to_mask]
        for col in cols:
            masked_H[H[:, col] != 0, col] = 0

    elif mode == 3:  # Randomly select non-zero elements and set them to zero
        non_zero_indices = torch.nonzero(H != 0)
        elements_to_mask = int(non_zero_indices.size(0) * ratio)
        indices = torch.randperm(non_zero_indices.size(0))[:elements_to_mask]
        for index in indices:
            row, col = non_zero_indices[index]
            masked_H[row, col] = 0

    return masked_H


if __name__ == '__main__':

    # 生成示例节点特征矩阵
    X = torch.rand(54, 64)  # 假设有100个节点，每个节点有100维特征

    # 构建超图的关联矩阵
    H = construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=True)

    # 根据关联矩阵生成超图的邻接矩阵
    G = generate_G_from_H(H)

    HG = adj2H(X, k_neigs=10)

    # 进行后续操作，例如超图的特征提取、分类等

    print('done')
