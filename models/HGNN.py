from torch import nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter

from models.HGNN_X2H_pt import *


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class HGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G))
        return x


class HGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(HGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x

class HGNN(nn.Module):
    def __init__(self, in_channels,num_electrodes, k_adj, out_channels, num_classes=2):
        super(HGNN, self).__init__()
        self.hgc1 = HGNN_conv(in_channels, out_channels)
        self.hgc2 = HGNN_conv(out_channels, out_channels)
        self.fc = Linear(num_electrodes*out_channels, num_classes)

    def forward(self, x, G):
        # H = construct_H_with_KNN(G, K_neigs=[10], split_diff_scale=True)
        # H = H[0]  # 输出的H有多个情况   参考超图原文

        # H_mask = self.random_mask_H_with_ratio(H, ratio=0.4, mode=1)
        # G = generate_G_from_H(H)

        x = F.relu(self.hgc1(x, G))
        # x = F.dropout(x, 0.5)
        result_emb = self.hgc2(x, G)
        result = result_emb.reshape(x.shape[0], -1)
        predict = self.fc(result)
        return predict, result_emb


if __name__ == '__main__':
    x = torch.randn(8, 54, 54)
    G = torch.randn(8, 54, 54)

    model = HGNN(54, 54, 64, 64)

    out = model(x, G)

    print(out[0].shape)
    print(out[1].shape)