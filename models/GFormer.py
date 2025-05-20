import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphTransformer_outch(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=2, num_layers=2):
        super(GraphTransformer_outch, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 使用线性层将输入从 in_channels 转换为 out_channels
        self.input_proj = nn.Linear(in_channels, out_channels)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=out_channels, nhead=num_heads, dim_feedforward=out_channels)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: (batch_size, num_nodes, in_channels)
        # 先通过线性层进行通道数转换
        x = self.input_proj(x)  # (batch_size, num_nodes, out_channels)
        x = x.permute(1, 0, 2)  # (num_nodes, batch_size, out_channels)

        for layer in self.layers:
            x = layer(x)

        return x.permute(1, 0, 2)  # (batch_size, num_nodes, out_channels)


# 定义 GraphTransformer
class GraphTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=2, num_layers=2):
        super(GraphTransformer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=in_channels, nhead=num_heads, dim_feedforward=out_channels)
            for _ in range(num_layers)
        ])

    def forward(self, x, L):
        # Assuming L is the adjacency matrix and x is the node features
        x = x.permute(1, 0, 2)  # (num_nodes, batch_size, in_channels)
        for layer in self.layers:
            x = layer(x)
        return x.permute(1, 0, 2)  # (batch_size, num_nodes, in_channels)

# 定义 DGCNN_encoder_moco
class DGCNN_encoder_moco_former(nn.Module):
    def __init__(self, in_channels, num_electrodes, k_adj, out_channels, num_classes=2, num_head=2, num_transformer_layers=2):
        super(DGCNN_encoder_moco_former, self).__init__()
        self.K = k_adj
        self.num_node = num_electrodes
        self.out_ch = out_channels
        self.layer1 = GraphTransformer_outch(in_channels, out_channels, num_heads=num_head, num_layers=num_transformer_layers)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = nn.Linear(num_electrodes * out_channels, 2)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes, num_electrodes).cuda())
        nn.init.uniform_(self.A, 0.01, 0.5)
        self.cl_fc = nn.Linear(num_electrodes * out_channels, out_channels)
        self.mask_node = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]

    def forward(self, x, if_cl=False):
        graph_emb = self.layer1(x)  # 注意这里传入的 x 和 L 需要符合 GraphTransformer 的输入要求
        graph_emb_view = graph_emb.reshape(graph_emb.size(0), -1)
        # graph_emb_view = graph_emb.reshape(self.num_node*self.out_ch, -1)
        # graph_emb_view = graph_emb.reshape(self.out_ch, -1)

        logits = self.fc(graph_emb_view)
        # logits = 0.

        if if_cl:
            cl_emb = self.cl_fc(graph_emb_view)
            emb = cl_emb
            # emb = graph_emb
        else:
            emb = graph_emb
        return logits, graph_emb, emb, emb


# 定义 normalize_A 函数
def normalize_A(A):
    # 对 A 进行归一化
    A = A + torch.eye(A.size(0)).cuda()  # 添加自环
    D = A.sum(dim=1)  # 度矩阵
    D_inv = D.pow(-1)  # 度矩阵的逆
    D_inv[torch.isinf(D_inv)] = 0  # 处理 D_inv 中的 Inf
    D_inv = torch.diag(D_inv)
    A_hat = torch.matmul(D_inv, A)
    return A_hat


if __name__ == '__main__':


    # 示例参数
    in_channels = 128  # 节点特征维度
    num_electrodes = 54  # 节点数量
    k_adj = 5  # Graph Transformer 中的邻接矩阵的 K 值
    out_channels = 128  # 图卷积的输出通道数
    num_classes = 2  # 分类任务中的类别数

    # 创建 DGCNN_encoder_moco 实例
    # model = DGCNN_encoder_moco(in_channels, num_electrodes, k_adj, out_channels, num_classes).cuda()
    model = DGCNN_encoder_moco_former(in_channels, num_electrodes, k_adj, out_channels, num_classes).cuda()

    # 创建随机的节点特征和邻接矩阵
    node_features = torch.randn(32, num_electrodes, in_channels).cuda()  # (batch_size, num_nodes, in_channels)
    adjacency_matrix = torch.randn(num_electrodes, num_electrodes).cuda()  # (num_nodes, num_nodes)

    # 测试模型前向传播
    logits, graph_emb, L, emb = model(node_features)

    print("Logits shape:", logits.shape)  # 应该是 (batch_size, num_classes)
    print("Graph embedding shape:", graph_emb.shape)  # 应该是 (batch_size, num_nodes, out_channels)
    print("L (normalized adjacency matrix) shape:", L.shape)  # 应该是 (num_nodes, num_nodes)
    print("Contrastive learning embedding shape:", emb.shape)  # 应该是 (batch_size, out_channels)
