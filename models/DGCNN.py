import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vq import VectorQuantizer
from models.mask_edge_strategy import random_mask_edge_mode
from models.tsne import tsne_graph
from models.HGNN_X2H_pt import *
from models.graph_prompt import SimplePrompt, GPFplusAtt
from models.graph_augmentation import *
from models.hypergraph_augmentation import *


def normalize_A(A,lmax=2):
    A=F.relu(A)
    N=A.shape[0]
    A=A*(torch.ones(N,N).cuda()-torch.eye(N,N).cuda())
    A=A+A.T
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    L = torch.eye(N,N).cuda()-torch.matmul(torch.matmul(D, A), D)
    Lnorm=(2*L/lmax)-torch.eye(N,N).cuda()
    return Lnorm


def generate_cheby_adj(L, K):
    device = L.device
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(L.shape[-1]).cuda())
        elif i == 1:
            support.append(L)
        else:
            temp = torch.matmul(2*L,support[-1].to(device),)-support[-2].to(device)
            support.append(temp)
    return support

class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels).cuda())
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).cuda())
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class Chebynet(nn.Module):
    def __init__(self, in_channels, K, out_channels):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc = nn.ModuleList()
        for i in range(K):
            self.gc.append(GraphConvolution(in_channels,  out_channels))

    def forward(self, x,L):
        adj = generate_cheby_adj(L, self.K)
        for i in range(len(self.gc)):
            if i == 0:
                result = self.gc[i](x, adj[i])
            else:
                result += self.gc[i](x, adj[i])
        result = F.relu(result)
        return result


class DGCNN(nn.Module):
    def __init__(self, in_channels,num_electrodes, k_adj, out_channels, num_classes=3):
        #in_channels(int): The feature dimension of each electrode.
        #num_electrodes(int): The number of electrodes.
        #k_adj(int): The number of graph convolutional layers.
        #out_channel(int): The feature dimension of  the graph after GCN.
        #num_classes(int): The number of classes to predict.
        super(DGCNN, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes*out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes).cuda())
        nn.init.uniform_(self.A,0.01,0.5)

    def forward(self, x):
        # x = self.BN1(x.transpose(1, 2)).transpose(1, 2) #data can also be standardized offline
        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = self.fc(result)
        return result



class DGCNN_L(nn.Module):
    def __init__(self, in_channels,num_electrodes, k_adj, out_channels, num_classes=3):
        #in_channels(int): The feature dimension of each electrode.
        #num_electrodes(int): The number of electrodes.
        #k_adj(int): The number of graph convolutional layers.
        #out_channel(int): The feature dimension of  the graph after GCN.
        #num_classes(int): The number of classes to predict.
        super(DGCNN_L, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes*out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes).cuda())
        nn.init.uniform_(self.A,0.01,0.5)

    def forward(self, x):
        # x = self.BN1(x.transpose(1, 2)).transpose(1, 2) #data can also be standardized offline
        L = normalize_A(self.A)
        emb = self.layer1(x, L)
        result = emb.reshape(x.shape[0], -1)
        logits = self.fc(result)
        return logits, emb, L

class DGCNN_prompt(nn.Module):
    def __init__(self, in_channels,num_electrodes, k_adj, out_channels, num_classes=3):
        #in_channels(int): The feature dimension of each electrode.
        #num_electrodes(int): The number of electrodes.
        #k_adj(int): The number of graph convolutional layers.
        #out_channel(int): The feature dimension of  the graph after GCN.
        #num_classes(int): The number of classes to predict.
        super(DGCNN_prompt, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes*out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes).cuda())
        nn.init.uniform_(self.A,0.01,0.5)

        self.mask_id = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]

        self.mask_embedding = nn.Parameter(torch.randn(out_channels))  # 可学习的mask向量

    def forward(self, x, adj, prompt):
        # x = self.BN1(x.transpose(1, 2)).transpose(1, 2) #data can also be standardized offline
        x = self.random_mask_node_mode(x, self.mask_id, n=30, mode=3)

        L = normalize_A(self.A)

        if prompt is not None:
            x = prompt.add(x)

        result_emb = self.layer1(x, L)

        result = result_emb.reshape(x.shape[0], -1)
        predict = self.fc(result)
        return predict, result_emb

    def random_mask_node_mode(self, input_features, masked_nodes, n, mode=1):
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
            # 在所有节点中随机选择n个节点   Default
            all_nodes = list(range(num_nodes))
            random_indices = torch.randperm(num_nodes)[:n]
            selected_nodes = [all_nodes[i] for i in random_indices]
        else:
            raise ValueError("Invalid mode. Mode must be 1, 2, or 3.")

        # 将选定的节点置为0
        # masked_features[:, selected_nodes, :] = self.mask_embedding
        masked_features[:, selected_nodes, :] = 0

        return masked_features



class DGCNN_prompt_joint(nn.Module):
    def __init__(self, in_channels,num_electrodes, k_adj, out_channels, mask_id, mask_ratio, mask_mode, num_classes=3):
        #in_channels(int): The feature dimension of each electrode.
        #num_electrodes(int): The number of electrodes.
        #k_adj(int): The number of graph convolutional layers.
        #out_channel(int): The feature dimension of  the graph after GCN.
        #num_classes(int): The number of classes to predict.
        super(DGCNN_prompt_joint, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes*out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes).cuda())
        nn.init.uniform_(self.A,0.01,0.5)

        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode

        self.mask_id = mask_id

        self.mask_embedding = nn.Parameter(torch.randn(out_channels))  # 可学习的mask向量

    def forward(self, x, adj, prompt):
        # x = self.BN1(x.transpose(1, 2)).transpose(1, 2) #data can also be standardized offline
        # x = self.random_mask_node_mode(x, self.mask_id, n=self.mask_ratio, mode=self.mask_mode)

        L = normalize_A(self.A)

        if prompt is not None:
            x = prompt.add(x)

        result_emb = self.layer1(x, L)

        result = result_emb.reshape(x.shape[0], -1)
        predict = self.fc(result)
        return predict, result_emb

    def random_mask_node_mode(self, input_features, masked_nodes, n, mode=1):
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
            # 在所有节点中随机选择n个节点   Default
            all_nodes = list(range(num_nodes))
            random_indices = torch.randperm(num_nodes)[:n]
            selected_nodes = [all_nodes[i] for i in random_indices]
        else:
            raise ValueError("Invalid mode. Mode must be 1, 2, or 3.")

        # 将选定的节点置为0
        masked_features[:, selected_nodes, :] = self.mask_embedding

        return masked_features

class DGCNN_encoder_mask(nn.Module):
    def __init__(self, in_channels,num_electrodes, k_adj, out_channels, mask_ratio, mask_mode, num_classes=2):
        super(DGCNN_encoder_mask, self).__init__()
        self.K = k_adj
        self.mask_mode = mask_mode
        self.mask_ratio = mask_ratio
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes*out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes).cuda())
        nn.init.uniform_(self.A,0.01,0.5)

        self.mask_id = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]

        self.mask_embedding = nn.Parameter(torch.randn(out_channels))  # 可学习的mask向量


    def forward(self, x, adj):
        # x = self.BN1(x.transpose(1, 2)).transpose(1, 2) #data can also be standardized offline

        x = self.random_mask_node_mode(x, self.mask_id, n=self.mask_ratio, mode=self.mask_mode)

        L = normalize_A(self.A)
        result_emb = self.layer1(x, L)

        result = result_emb.reshape(x.shape[0], -1)
        predict = self.fc(result)
        return predict, result_emb

    def random_mask_node_mode(self, input_features, masked_nodes, n, mode=1):
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
            # 在所有节点中随机选择n个节点   Default
            all_nodes = list(range(num_nodes))
            random_indices = torch.randperm(num_nodes)[:n]
            selected_nodes = [all_nodes[i] for i in random_indices]
        else:
            raise ValueError("Invalid mode. Mode must be 1, 2, or 3.")

        # 将选定的节点置为0
        masked_features[:, selected_nodes, :] = self.mask_embedding
        # masked_features[:, selected_nodes, :] = 0

        return masked_features



class DGCNN_hyper_encoder_mask(nn.Module):
    def __init__(self, in_channels,num_electrodes, k_adj, out_channels, num_classes=2):
        super(DGCNN_hyper_encoder_mask, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes*out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes).cuda())
        nn.init.uniform_(self.A,0.01,0.5)

        self.mask_id = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]

        self.mask_embedding = nn.Parameter(torch.randn(out_channels))  # 可学习的mask向量


    def forward(self, x, adj):
        # x = self.BN1(x.transpose(1, 2)).transpose(1, 2) #data can also be standardized offline

        x = self.random_mask_node_mode(x, self.mask_id, n=30, mode=3)

        L = normalize_A(self.A)

        # H = construct_H_with_KNN(L, K_neigs=[10], split_diff_scale=True)
        # H = H[0]   #输出的H有多个情况   参考超图原文
        # # H = self.random_mask_H_with_ratio(H, ratio=0.3, mode=3)
        # G = generate_G_from_H(H, variable_weight=True)
        # result_emb = self.layer1(x, G[1].cuda())

        result_emb = self.layer1(x, L)

        result = result_emb.reshape(x.shape[0], -1)
        predict = self.fc(result)
        return predict, result_emb

    def random_mask_H_with_ratio(self, H, ratio, mode):
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

    def random_mask_node_mode(self, input_features, masked_nodes, n, mode=1):
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
            # 在所有节点中随机选择n个节点   Default
            all_nodes = list(range(num_nodes))
            random_indices = torch.randperm(num_nodes)[:n]
            selected_nodes = [all_nodes[i] for i in random_indices]
        else:
            raise ValueError("Invalid mode. Mode must be 1, 2, or 3.")

        # 将选定的节点置为0
        masked_features[:, selected_nodes, :] = self.mask_embedding
        # masked_features[:, selected_nodes, :] = 0

        return masked_features




class DGCNN_hyper_encoder_mask_joint(nn.Module):
    def __init__(self, in_channels, num_electrodes, k_adj, out_channels, num_classes=2):
        super(DGCNN_hyper_encoder_mask_joint, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes * out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes).cuda())
        nn.init.uniform_(self.A, 0.01, 0.5)

        self.mask_id = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]
        self.mask_id_s = list(range(24))

        self.mask_embedding = nn.Parameter(torch.randn(out_channels))  # 可学习的mask向量

    def forward(self, x, adj):
        # x = self.BN1(x.transpose(1, 2)).transpose(1, 2) #data can also be standardized offline

        x = self.random_mask_node_mode(x, self.mask_id, n=30, mode=3)

        L = normalize_A(self.A)

        H = construct_H_with_KNN(L, K_neigs=[10], split_diff_scale=True)
        H = H[0]   #输出的H有多个情况   参考超图原文
        H = self.random_mask_H_with_ratio(H, ratio=0.3, mode=3)
        G = generate_G_from_H(H, variable_weight=True)
        result_emb = self.layer1(x, G[1].cuda())

        # result_emb = self.layer1(x, L)

        result = result_emb.reshape(x.shape[0], -1)
        predict = self.fc(result)
        return predict, result_emb

    def random_mask_H_with_ratio(self, H, ratio, mode):
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


    def random_mask_node_mode(self, input_features, masked_nodes, n, mode=1):
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
            # 在所有节点中随机选择n个节点   Default
            all_nodes = list(range(num_nodes))
            random_indices = torch.randperm(num_nodes)[:n]
            selected_nodes = [all_nodes[i] for i in random_indices]
        else:
            raise ValueError("Invalid mode. Mode must be 1, 2, or 3.")

        # 将选定的节点置为0
        masked_features[:, selected_nodes, :] = self.mask_embedding

        return masked_features


class DGCNN_encoder(nn.Module):
    def __init__(self, in_channels,num_electrodes, k_adj, out_channels, num_classes=2):
        super(DGCNN_encoder, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes*out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes).cuda())
        nn.init.uniform_(self.A,0.01,0.5)

    def forward(self, x):
        # x = self.BN1(x.transpose(1, 2)).transpose(1, 2) #data can also be standardized offline
        L = normalize_A(self.A)
        result_emb = self.layer1(x, L)

        result = result_emb.reshape(x.shape[0], -1)
        predict = self.fc(result)
        return predict, result_emb


class DGCNN_encoder_reg(nn.Module):
    def __init__(self, in_channels, num_electrodes, k_adj, out_channels):
        super(DGCNN_encoder_reg, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes * out_channels, 1)  # Change output to 1 for regression
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes).cuda())
        nn.init.uniform_(self.A, 0.01, 0.5)

    def forward(self, x):
        # x = self.BN1(x.transpose(1, 2)).transpose(1, 2) #data can also be standardized offline
        L = normalize_A(self.A)
        result_emb = self.layer1(x, L)

        result = result_emb.reshape(x.shape[0], -1)
        predict = self.fc(result)
        return predict.squeeze(), result_emb


class DGCNN_encoder_mask_edge(nn.Module):
    def __init__(self, in_channels, num_electrodes, k_adj, out_channels, num_classes=2, mask_mode=3):
        super(DGCNN_encoder_mask_edge, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes*out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes).cuda())
        nn.init.uniform_(self.A,0.01,0.5)
        self.mask_mode = mask_mode
        self.mask_node = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]

    def forward(self, x, adj):
        # x = self.BN1(x.transpose(1, 2)).transpose(1, 2) #data can also be standardized offline
        L = normalize_A(self.A)
        # L = normalize_A(self.A)
        # L = random_mask_edge_mode(L, self.mask_node, mode=self.mask_mode, ratio=0.0)

        result_emb = self.layer1(x, L)

        result = result_emb.reshape(x.shape[0], -1)
        predict = self.fc(result)
        return predict, result_emb


class DGCNN_encoder_cl(nn.Module):
    def __init__(self, in_channels, num_electrodes, k_adj, out_channels, num_classes=2, mask_mode=3):
        super(DGCNN_encoder_cl, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.cl_fc = Linear(num_electrodes*out_channels, 128)
        self.fc = Linear(num_electrodes*out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes).cuda())
        nn.init.uniform_(self.A,0.01,0.5)
        self.mask_mode = mask_mode
        self.mask_node = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]

    def forward(self, x):

        L = normalize_A(self.A)


        result_emb = self.layer1(x, L)

        result = result_emb.reshape(x.shape[0], -1)
        emb = self.cl_fc(result)
        predict = self.fc(result)
        return predict, emb


class DGCNN_encoder_moco(nn.Module):
    def __init__(self, in_channels, num_electrodes, k_adj, out_channels, num_classes, num_layers):
        super(DGCNN_encoder_moco, self).__init__()
        self.K = k_adj
        self.num_layers = num_layers

        # 创建多个Chebynet层
        self.layers = nn.ModuleList(
            [Chebynet(in_channels if i == 0 else out_channels, k_adj, out_channels) for i in range(num_layers)])
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes * out_channels, num_classes)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes).cuda())
        nn.init.uniform_(self.A, 0.01, 0.5)
        self.cl_fc = Linear(num_electrodes * out_channels, out_channels)
        # self.mask_mode = mask_mode
        self.mask_node = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]

    def forward(self, x, if_cl=True):

        L = normalize_A(self.A)

        # 通过所有Chebynet层
        for layer in self.layers:
            x = layer(x, L)

        graph_emb = x
        graph_emb_view = graph_emb.view(graph_emb.size(0), -1)
        cl_emb = self.cl_fc(graph_emb_view)
        logits = self.fc(graph_emb_view)

        if if_cl:
            emb = cl_emb
        else:
            emb = graph_emb
        return logits, graph_emb, L, emb


class DGCNN_encoder_moco_fmri(nn.Module):
    def __init__(self, in_channels, num_electrodes, k_adj, out_channels, num_classes):
        super(DGCNN_encoder_moco_fmri, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes * out_channels, num_classes)
        self.cl_fc = Linear(num_electrodes * out_channels, out_channels)
        self.mask_node = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]

    def forward(self, node_features, adj, if_cl=True):
        # Normalize adjacency matrix
        L = normalize_A(adj)

        # Apply graph convolution
        graph_emb = self.layer1(node_features, L)
        graph_emb_view = graph_emb.view(graph_emb.size(0), -1)

        # Classification and contrastive learning embeddings
        cl_emb = self.cl_fc(graph_emb_view)
        logits = self.fc(graph_emb_view)

        if if_cl:
            emb = cl_emb
        else:
            emb = graph_emb

        return logits, graph_emb, L, emb


class DGCNN_encoder_moco_dynamic(nn.Module):
    def __init__(self, in_channels, k_adj, out_channels, num_classes):
        super(DGCNN_encoder_moco_dynamic, self).__init__()
        self.K = k_adj
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = None  # 初始化时不设置输入维度
        self.cl_fc = None  # 初始化时不设置输入维度

        # 初始化可学习的邻接矩阵 A
        self.A = None
        self.mask_node = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]

    def initialize_parameters(self, num_electrodes, device):
        # 重新创建 Linear 层
        self.fc = nn.Linear(num_electrodes * self.out_channels, self.num_classes).to(device)
        self.cl_fc = nn.Linear(num_electrodes * self.out_channels, self.out_channels).to(device)

        # 初始化 A 矩阵为一个可以学习的参数
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes).to(device))
        nn.init.xavier_uniform_(self.A, gain=1.0)  # 初始化为 Xavier 分布

    def forward(self, x, if_cl=True):
        device = x.device  # 获取输入张量所在的设备

        num_electrodes = x.size(1)
        if self.fc is None or self.cl_fc is None:
            self.initialize_parameters(num_electrodes, device)

        # 生成邻接矩阵 A
        A = self.generate_adjacency_matrix(x)

        # 计算拉普拉斯矩阵 L
        L = normalize_A(A.to(device))

        # 图卷积层
        graph_emb = self.layer1(x, L)
        graph_emb_view = graph_emb.view(graph_emb.size(0), -1)
        cl_emb = self.cl_fc(graph_emb_view)
        logits = self.fc(graph_emb_view)

        if if_cl:
            emb = cl_emb
        else:
            emb = graph_emb
        return logits, graph_emb, L, emb

    def generate_adjacency_matrix(self, x):
        num_electrodes = x.size(1)
        # 生成邻接矩阵 A
        A = torch.sigmoid(self.A)  # 通过 sigmoid 函数将 A 限制在 0 到 1 之间
        A = (A + A.T) / 2  # 确保 A 是对称的
        A.fill_diagonal_(0)  # 对角线元素设为0，表示自环

        # 取 K 个最近邻
        A_binary = torch.zeros_like(A)
        for i in range(num_electrodes):
            _, indices = A[i].topk(self.K, largest=True, sorted=False)
            A_binary[i, indices] = 1

        return A_binary


class DGCNN_decoder(nn.Module):
    def __init__(self, in_channels, num_electrodes, k_adj, out_channels):
        super(DGCNN_decoder, self).__init__()
        self.K = k_adj
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes).cuda())
        nn.init.uniform_(self.A,0.01,0.5)

    def forward(self, x):

        L = normalize_A(self.A)

        graph_emb = self.layer1(x, L)
        return graph_emb


class Graph_decoder(nn.Module):
    def __init__(self, in_features, num_nodes, out_features):
        super(Graph_decoder, self).__init__()
        self.fc1 = nn.Linear(in_features, num_nodes)  # Map to the number of nodes
        self.A = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))  # Adjacency matrix
        nn.init.uniform_(self.A, 0.01, 0.5)
        self.fc2 = nn.Linear(num_nodes, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.matmul(x, self.A)  # Apply the adjacency matrix
        x = self.fc2(x)
        return x




if __name__ == '__main__':

    # 定义输入数据
    batch_size = 256
    num_channels = 64
    num_nodes = 54
    num_classes = 3

    # 生成示例输入数据
    x = torch.randn(batch_size, num_nodes, num_channels)  # 输入数据，形状为 (batch_size, num_channels, num_nodes)
    adj = torch.randn(batch_size, num_channels, num_channels)  # 输入数据，形状为 (batch_size, num_channels, num_nodes)
    # L = torch.randn(num_nodes, num_nodes)  # 图 Laplacian 矩阵，形状为 (num_nodes, num_nodes)

    # 实例化模型
    model = DGCNN_hyper_encoder_mask(in_channels=num_channels,
                  num_electrodes=num_nodes,
                  k_adj=3,
                  out_channels=64,
                  num_classes=num_classes)

    model_de = Graph_decoder(64, 128, 64)

    # 将模型移动到 GPU 上（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_de = model_de.to(device)
    x = x.to(device)
    # L = L.to(device)

    # 使用模型进行前向传播
    output, emb = model(x, adj)
    re_emb = model_de(emb)

    # tsne_graph(emb.reshape(x.shape[0], -1))

    # 输出结果
    print("模型输出的形状:", output.shape)
