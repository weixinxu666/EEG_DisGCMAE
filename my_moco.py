import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.graph_augmentation import graph_aug_feat_moco
from DGCNN import DGCNN_encoder_moco, Graph_decoder, DGCNN_encoder_moco_dynamic
from loss_cl import NTXentLoss_poly, NTXentLoss_poly_batch

from torch.backends import cudnn

device = torch.device("cuda:0")
cudnn.benchmark = True
cudnn.benchmark = False
cudnn.deterministic = True


class MoCo(nn.Module):
    def __init__(self, gcn_encoder_t, gcn_encoder_s, decoder_t, decoder_s, queue_size, out_channels, momentum, temperature):
        super(MoCo, self).__init__()
        self.encoder_q_t = gcn_encoder_t
        self.encoder_q_s = gcn_encoder_s
        self.encoder_k_t = gcn_encoder_t
        self.encoder_k_s = gcn_encoder_s
        self.decoder_t = decoder_t
        self.decoder_s = decoder_s
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.register_buffer("queue", torch.randn(queue_size, out_channels))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.queue_ptr = 0

        self.mse_loss = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        # self.cl_loss = NTXentLoss_poly_batch(device=device, temperature=0.2, use_cosine_similarity=True)

    def forward(self, x_t, x_s):
        x_q_t, x_k_t, x_q_t_mask, x_k_t_mask = self.graph_aug_feat_moco(x_t, drop_ratio=0.)
        x_q_s, x_k_s, x_q_s_mask, x_k_s_mask = self.graph_aug_feat_moco(x_s, drop_ratio=0.)

        # x_q_t_mask, x_k_t_mask = x_q_t, x_k_t
        # x_q_s_mask, x_k_s_mask = x_q_s, x_k_s

        _, x_m_e_q_t, _, _ = self.encoder_q_t(x_q_t_mask, False)
        _, x_m_e_k_t, _, _ = self.encoder_q_t(x_k_t_mask, False)
        _, x_m_e_q_s, _, _ = self.encoder_q_s(x_q_s_mask, False)
        _, x_m_e_k_s, _, _ = self.encoder_q_s(x_k_s_mask, False)

        x_rec_t = self.decoder_t(torch.cat([x_m_e_q_t, x_m_e_k_t], dim=0))
        x_rec_s = self.decoder_s(torch.cat([x_m_e_q_s, x_m_e_k_s], dim=0))

        loss_rec_t = self.reconstruct_loss(x_rec_t, torch.cat([x_t, x_t], dim=0))
        loss_rec_s = self.reconstruct_loss(x_rec_s, torch.cat([x_s, x_s], dim=0))
        loss_rec = loss_rec_t + loss_rec_s

        x_k_t_mix = torch.cat([x_k_t, x_rec_t], dim=0)
        x_k_s_mix = torch.cat([x_k_s, x_rec_s], dim=0)

        _, q_t, _, q_t_readout = self.encoder_q_t(x_q_t, True)
        _, q_s, _, q_s_readout = self.encoder_q_s(x_q_s, True)
        _, k_t, _, k_t_readout = self.encoder_k_t(x_k_t_mix, True)
        _, k_s, _, k_s_readout = self.encoder_k_s(x_k_s_mix, True)
        k_mix = torch.cat([k_t_readout, k_s_readout], dim=0)
        k_mix = k_mix.detach()

        pos_t = torch.bmm(q_t_readout.view(-1, 1, q_t_readout.size(-1)),
                          k_mix.view(-1, k_t_readout.size(-1), 6)).squeeze(-1)
        neg_t = torch.mm(q_t_readout, self.queue.T)
        pos_s = torch.bmm(q_s_readout.view(-1, 1, q_s_readout.size(-1)), k_mix.view(-1, k_mix.size(-1), 6)).squeeze(-1)
        neg_s = torch.mm(q_s_readout, self.queue.T)

        logits_t = torch.cat([pos_t.squeeze(), neg_t], dim=1)
        logits_s = torch.cat([pos_s.squeeze(), neg_s], dim=1)

        labels = torch.zeros(logits_t.size(0), dtype=torch.long).to(logits_t.device)
        loss_cl_t = F.cross_entropy(logits_t / self.temperature, labels)
        loss_cl_s = F.cross_entropy(logits_s / self.temperature, labels)
        loss_cl = loss_cl_t + loss_cl_s

        loss = loss_rec + loss_cl
        return loss, loss_rec_t, loss_rec_s, loss_cl_t, loss_cl_s, x_rec_t, x_rec_s

    def update_queue(self, k):
        batch_size = k.size(0)
        # print(batch_size, self.queue.size(0))
        assert batch_size == self.queue.size(0), "Batch size must match the queue size"
        ptr = int(self.queue_ptr)
        self.queue[ptr:ptr + batch_size, :] = k  # Replace old queue entries with new entries
        ptr = (ptr + batch_size) % self.queue_size  # Update the pointer to the next position
        self.queue_ptr = ptr

    def remove_nodes_and_add_noise(self, node_features, drop_ratio, noise_level, seed=None):
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

    def replace_nodes_with_embedding(self, node_features, indices):
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

    def graph_aug_feat_moco(self, node_features, drop_ratio=0.2, noise_level=0.2):
        """
        添加噪声并移除节点（对 query 和 key）
        """
        if node_features is None:
            raise ValueError("node_features cannot be None")

        query = self.remove_nodes_and_add_noise(node_features.clone(), drop_ratio, noise_level, seed=88)
        key = self.remove_nodes_and_add_noise(node_features.clone(), drop_ratio, noise_level, seed=32)

        # 生成随机索引
        num_nodes = node_features.size(1)
        indices_query = torch.randint(0, num_nodes, (int(drop_ratio * num_nodes),), device=node_features.device)
        indices_key = torch.randint(0, num_nodes, (int(drop_ratio * num_nodes),), device=node_features.device)

        # 将移除的节点用嵌入向量替换（对 query）
        query_emb = self.replace_nodes_with_embedding(query.clone(), indices_query)

        # 将移除的节点用嵌入向量替换（对 key）
        key_emb = self.replace_nodes_with_embedding(key.clone(), indices_key)

        return query, key, query_emb, key_emb

    def mix(self, x1, x2):
        # 实现你的混合方法
        pass

    def embedding_insert(self, x):
        # 实现你的嵌入方法
        pass

    def reconstruct_loss(self, x_rec, x):
        loss = self.mse_loss(x, x_rec)
        return loss

    def distill_loss(self, x_rec_t, A_rec_t, x_rec_s, A_rec_s):
        # 实现你的蒸馏损失方法
        pass


if __name__ == '__main__':
    import torch
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader


    # 定义一个简单的数据集
    class SimpleDataset(Dataset):
        def __init__(self, num_samples, num_nodes_t, num_nodes_s, num_features):
            self.data_t = torch.randn(num_samples, num_nodes_t, num_features)  # 图1
            self.data_s = torch.randn(num_samples, num_nodes_s, num_features)  # 图2
            self.labels = torch.randint(0, 2, (num_samples,))  # 示例中的标签是0或1

        def __len__(self):
            return len(self.data_t)

        def __getitem__(self, idx):
            return self.data_t[idx], self.data_s[idx], self.labels[idx]


    # 初始化模型和数据加载器
    gcn_encoder_t = DGCNN_encoder_moco(in_channels=128, num_electrodes=54, k_adj=10, out_channels=128, num_classes=2)
    gcn_encoder_s = DGCNN_encoder_moco(in_channels=128, num_electrodes=24, k_adj=10, out_channels=128, num_classes=2)
    decoder_t = Graph_decoder(in_features=128, num_nodes=54, out_features=128)
    decoder_s = Graph_decoder(in_features=128, num_nodes=24, out_features=128)
    queue_size = 64
    momentum = 0.999
    temperature = 15.
    out_channels = 128
    model = MoCo(gcn_encoder_t, gcn_encoder_s, decoder_t, decoder_s, queue_size, out_channels, momentum, temperature)
    model = model.cuda()

    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 创建数据加载器
    dataset = SimpleDataset(num_samples=500, num_nodes_t=54, num_nodes_s=24, num_features=128)  # 根据实际情况设置数据
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 训练循环
    num_epochs = 20  # 设置训练的轮数

    for epoch in range(num_epochs):
        model.train()
        for x_t, x_s, _ in data_loader:
            x_t, x_s = x_t.cuda(), x_s.cuda()

            optimizer.zero_grad()
            # 前向传播
            loss, loss_rec_t, loss_rec_s, loss_cl_t, loss_cl_s, _, _ = model(x_t, x_s)

            # 反向传播和优化
            # 如果需要多次反向传播，可以设置 retain_graph=True
            loss.backward(retain_graph=True)  # 保留计算图

            optimizer.step()

            # 更新动量编码器的参数
            for param_q, param_k in zip(model.encoder_q_t.parameters(), model.encoder_k_t.parameters()):
                param_k.data = model.momentum * param_k.data + (1 - model.momentum) * param_q.data
            for param_q, param_k in zip(model.encoder_q_s.parameters(), model.encoder_k_s.parameters()):
                param_k.data = model.momentum * param_k.data + (1 - model.momentum) * param_q.data

            # 生成 k_mix 用于更新队列
            x_k_t_mix = torch.cat([x_t, torch.randn_like(x_t)], dim=0)  # 生成额外的 `x_k_t`
            x_k_s_mix = torch.cat([x_s, torch.randn_like(x_s)], dim=0)  # 生成额外的 `x_k_s`

            _, _, _, k_t_readout = model.encoder_k_t(x_k_t_mix, True)
            _, _, _, k_s_readout = model.encoder_k_s(x_k_s_mix, True)
            k_mix = torch.cat([k_t_readout, k_s_readout], dim=0)  # 计算 `k_mix` 为对比学习的负样本

            # 确保 k_mix 的维度是 [queue_size, out_channels]
            k_mix = k_mix[:queue_size, :]  # 修剪 k_mix 以匹配队列的大小

            model.update_queue(k_mix)

            # 打印训练信息
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Rec_T Loss: {loss_rec_t.item():.4f}, Rec_S Loss: {loss_rec_s.item():.4f}, CL_T Loss: {loss_cl_t.item():.4f}, CL_S Loss: {loss_cl_s.item():.4f}')
