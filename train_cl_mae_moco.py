import argparse
import os

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

from torch.utils.data import DataLoader
from torch.autograd import Variable

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from DGCNN import *
from GCN import *
from tools.HGNN_X2H import *

# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from loss import *

from models.vq_graph import VectorQuantizer
from models.mask_strategy import random_mask_node, random_mask_node_mode
from models.tsne import tsne_visual, tsne_visual_3d
from models.HGNN import *
from models.graph_augmentation import *
from models.hypergraph_augmentation import *
from loss_cl import NTXentLoss_poly, NTXentLoss_poly_batch
from models.my_moco import MoCo

device = torch.device("cuda:0")
cudnn.benchmark = True


cudnn.benchmark = False
cudnn.deterministic = True


class ExP():
    def __init__(self, nsub):
        super(ExP, self).__init__()
        self.batch_size = 64
        self.n_epochs = 7
        self.c_dim = 4
        self.lr = 0.002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.nSub = nsub

        self.start_epoch = 0
        self.root = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/EMBARC/'
        self.save_path = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/exp_results/cl_mae/'

        # self.log_write = open(os.path.join(self.save_path, 'performance.txt'), "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.mse_loss = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.node_num_t = 54
        self.node_num_s = 31

        self.feat_num = 128

        # EMBARC 8
        # self.mask_id = [30, 31, 17, 18, 4, 5, 38, 39]
        # EMBARC 24
        # self.mask_id = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]
        # HBN 32C
        # self.mask_id = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125]


        # self.mask_id = [30, 31, 17, 18, 4, 5, 38, 39]  # 8 channels
        # self.mask_id = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]
        # 32sys
        # self.mask_id = [33, 34, 32, 17, 18, 15, 16, 24, 25, 28, 29, 10, 11, 43, 44, 8, 9, 12, 13, 38, 39, 41, 42, 43, 46, 47, 40, 31, 32, 41, 37, 38]
        # 16 sys
        # self.mask_id = [33, 34, 32, 17, 18, 15, 16, 24, 25, 10, 11, 43, 44, 38, 39, 40]
        #31 sys
        self.mask_id = [0, 1, 4, 5, 8, 9, 12, 13, 17, 18, 21, 22, 23, 24, 27, 28, 30, 31, 32, 33, 34, 35, 38, 39, 42, 43, 44, 45, 49, 50, 51]
        #12 sys
        # self.mask_id = [4, 5, 17, 18, 30, 31, 33, 34, 38, 39, 50, 51]


        self.gcn_encoder_t = DGCNN_encoder_moco(in_channels=128, num_electrodes=self.node_num_t, k_adj=10, out_channels=128,
                                           num_classes=2)
        self.gcn_encoder_s = DGCNN_encoder_moco(in_channels=128, num_electrodes=self.node_num_s, k_adj=10, out_channels=128,
                                           num_classes=2)
        self.decoder_t = Graph_decoder(in_features=128, num_nodes=self.node_num_t, out_features=128)
        self.decoder_s = Graph_decoder(in_features=128, num_nodes=self.node_num_s, out_features=128)
        self.queue_size = self.batch_size
        self.momentum = 0.999
        self.temperature = 15.0
        self.out_channels = 128
        self.model = MoCo(self.gcn_encoder_t, self.gcn_encoder_s, self.decoder_t, self.decoder_s, self.queue_size, self.out_channels, self.momentum,
                     self.temperature)
        self.model = self.model.cuda()


    def get_embarc_graph_data(self, path):

        self.data = np.load(path, allow_pickle=True).item()

        eeg_ts_data = self.data["eeg_ts"]
        eeg_corr_data = self.data["eeg_corr"]
        labels = self.data["labels"]

        # final_incidence = []
        # for i in range(eeg_corr_data.shape[0]):
        #     h = adj2H(eeg_corr_data[i, :, :], k_neigs=5)
        #     final_incidence.append(h)
        # eeg_corr_data = np.array(eeg_corr_data)

        # scaler = StandardScaler()
        # self.train_data = scaler.fit_transform(self.train_data)
        # self.test_data = scaler.transform(self.test_data)

        # standardize
        target_mean = np.mean(eeg_ts_data)
        target_std = np.std(eeg_ts_data)
        eeg_ts_data = (eeg_ts_data - target_mean) / target_std
        eeg_ts_data = (eeg_ts_data - target_mean) / target_std

        target_mean = np.mean(eeg_corr_data)
        target_std = np.std(eeg_corr_data)
        eeg_corr_data = (eeg_corr_data - target_mean) / target_std

        # X_combined = np.hstack((eeg_ts_data, eeg_corr_data))

        # # permute labels
        # perm = np.random.permutation(len(labels))
        # labels = labels[perm]

        # return X_combined, labels
        return eeg_ts_data, labels


    def train(self, data_t, data_s):

        train_ts_t, test_ts_t, train_label_t, test_label_t = data_t[0], data_t[1], data_t[2], data_t[3]
        train_ts_s, test_ts_s, train_label_s, test_label_s = data_s[0], data_s[1], data_s[2], data_s[3]


        train_label = train_label_t.squeeze()
        test_label = test_label_t.squeeze()

        node_feat_train_t, node_feat_train_s = torch.from_numpy(train_ts_t), torch.from_numpy(train_ts_s)
        # edge_matrix_train_t, edge_matrix_train_s = torch.from_numpy(train_corr_t), torch.from_numpy(train_corr_s)
        train_label = torch.from_numpy(train_label)

        train_label = torch.tensor(train_label, dtype=torch.int64)
        test_label = torch.tensor(test_label, dtype=torch.int64)

        dataset = torch.utils.data.TensorDataset(node_feat_train_t, node_feat_train_s, train_label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        node_feat_test_t, node_feat_test_s = torch.from_numpy(test_ts_t), torch.from_numpy(test_ts_s)
        # edge_matrix_test_t, edge_matrix_test_s = torch.from_numpy(test_corr_t), torch.from_numpy(test_corr_s)
        # test_dataset = torch.utils.data.TensorDataset(node_feat_test_t, node_feat_test_s, edge_matrix_test_t, edge_matrix_test_s, test_label)
        # self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size,
        #                                                    shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        node_feat_test_t, node_feat_test_s = Variable(node_feat_test_t.type(self.Tensor)), Variable(node_feat_test_s.type(self.Tensor))
        # edge_matrix_test_t, edge_matrix_test_s = Variable(edge_matrix_test_t.type(self.Tensor)), Variable(edge_matrix_test_s.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))


        loss_npy = []
        loss_cl_t_npy = []
        loss_cl_s_npy = []
        loss_rec_t_npy = []
        loss_rec_s_npy = []

        x_rec_t_list = []
        x_rec_s_list = []


        # saved_state_dict = torch.load(
        #     '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/exp_results/embarc_54eeg_graph_sex/model_0.839.pth')
        # self.model_t.load_state_dict(saved_state_dict)

        for e in range(self.n_epochs):

            self.model.train()
            for i, (node_feat_train_t, node_feat_train_s, train_label) in enumerate(self.dataloader):
                node_feat_train_t, node_feat_train_s = Variable(node_feat_train_t.cuda().type(self.Tensor)), Variable(node_feat_train_s.cuda().type(self.Tensor))
                # edge_matrix_train_t, edge_matrix_train_s = Variable(edge_matrix_train_t.cuda().type(self.Tensor)), Variable(edge_matrix_train_s.cuda().type(self.Tensor))

                # saved_state_dict = torch.load('/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/exp_results/embarc_54eeg_graph_sex/model_0.839.pth')
                # self.model_t.load_state_dict(saved_state_dict)

                x_t, x_s = node_feat_train_t, node_feat_train_s

                loss, loss_rec_t, loss_rec_s, loss_cl_t, loss_cl_s, x_rec_t, x_rec_s = self.model(x_t, x_s)


                self.optimizer.zero_grad()
                # 反向传播和优化
                # 如果需要多次反向传播，可以设置 retain_graph=True
                loss.backward(retain_graph=True)  # 保留计算图
                self.optimizer.step()

                for param_q, param_k in zip(self.model.encoder_q_t.parameters(), self.model.encoder_k_t.parameters()):
                    param_k.data = self.model.momentum * param_k.data + (1 - self.model.momentum) * param_q.data
                for param_q, param_k in zip(self.model.encoder_q_s.parameters(), self.model.encoder_k_s.parameters()):
                    param_k.data = self.model.momentum * param_k.data + (1 - self.model.momentum) * param_q.data

                # 生成 k_mix 用于更新队列
                x_k_t_mix = torch.cat([x_t, torch.randn_like(x_t)], dim=0)  # 生成额外的 `x_k_t`
                x_k_s_mix = torch.cat([x_s, torch.randn_like(x_s)], dim=0)  # 生成额外的 `x_k_s`

                _, _, _, k_t_readout = self.model.encoder_k_t(x_k_t_mix, True)
                _, _, _, k_s_readout = self.model.encoder_k_s(x_k_s_mix, True)
                k_mix = torch.cat([k_t_readout, k_s_readout], dim=0)  # 计算 `k_mix` 为对比学习的负样本

                # 确保 k_mix 的维度是 [queue_size, out_channels]
                k_mix = k_mix[:self.queue_size, :]  # 修剪 k_mix 以匹配队列的大小

                self.model.update_queue(k_mix)

                # print('Epoch: ', e, '####Total: ', losses.item(), '####MSE1 losses: ', loss_mse1.item(), '####VQ losses: ', loss_vq.item(), '####Perp: {}'.format(perplexity.item()))
                print(
                    f'Epoch: {e}\tTotal loss: {loss.item():.4f}\tTeacher rec loss: {loss_rec_t.item():.4f}\tStudent rec loss: {loss_rec_s.item():.4f}\tTeacher CL loss: {loss_cl_t.item():.4f}\tstudent CL loss: {loss_cl_s.item():.4f}')
                #
                # print('Epoch: ', e, '####Total: ', loss.item(), '####rec_t: ', loss_rec_t, '####rec_s: ', loss_rec_s, '####cl_t: ', loss_cl_t, '####cl_s: ',  loss_cl_s)

                loss_npy.append(loss.item())
                loss_cl_t_npy.append(loss_cl_t.item())
                loss_cl_s_npy.append(loss_cl_s.item())
                loss_rec_t_npy.append(loss_rec_t.item())
                loss_rec_s_npy.append(loss_rec_s.item())

                x_rec_t_list.append(x_rec_t)
                x_rec_s_list.append(x_rec_s)






                # torch.save(self.model.encoder_q_t.state_dict(), os.path.join(self.save_path, 't.pth'))
                # torch.save(self.model.encoder_q_s.state_dict(), os.path.join(self.save_path, 's.pth'))

        np.save(os.path.join(self.save_path, 'total_loss.npy'), np.array(loss_npy))
        np.save(os.path.join(self.save_path, 'loss_cl_t.npy'), np.array(loss_cl_t_npy))
        np.save(os.path.join(self.save_path, 'loss_cl_s.npy'), np.array(loss_cl_s_npy))
        np.save(os.path.join(self.save_path, 'loss_rec_t.npy'), np.array(loss_rec_t_npy))
        np.save(os.path.join(self.save_path, 'loss_rec_s.npy'), np.array(loss_rec_s_npy))

        # x_rec_t_ = torch.cat(x_rec_t_list, dim=0)
        # x_rec_s_ = torch.cat(x_rec_s_list, dim=0)

        # np.save(os.path.join(self.save_path, 'rec_t.npy'), np.array(x_rec_t_.cpu().detach()))
        # np.save(os.path.join(self.save_path, 'rec_t3.npy'), np.array(x_rec_t_.cpu().detach()))
        # np.save(os.path.join(self.save_path, 'rec_tt.npy'), np.array(x_rec_t_.cpu().detach()))
        # np.save(os.path.join(self.save_path, 'rec_s.npy'), np.array(x_rec_s_.cpu().detach()))
        # np.save(os.path.join(self.save_path, 'rec_s3.npy'), np.array(x_rec_s_.cpu().detach()))



        return loss.item()



