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
from models.MLP import MLPClassifier
from losses.gnn2gnn2 import *
from losses.gnn2mlp2 import *
from models.GFormer import *
from models.share_gnn import DynamicEdgeGNN

cudnn.benchmark = False
cudnn.deterministic = True


class ExP():
    def __init__(self, nsub):
        super(ExP, self).__init__()
        self.batch_size = 64
        self.n_epochs = 400
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.nSub = nsub

        self.start_epoch = 0
        self.root = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/EMBARC/'
        self.save_path = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/exp_results/distill/'

        # self.log_write = open(os.path.join(self.save_path, 'performance.txt'), "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.mse_loss = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.lsp_loss = LSP_Loss(kernel_type='linear', threshold=0.3).cuda()    #poly崩了   linear最好

        # self.mask_id = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]
        # HBN 32C
        self.mask_id = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125]
        #HBN 64C
        # self.mask_id = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127]

        self.node_num_t = 54
        self.node_num_s = 24

        self.feat_num = 128

        # self.model_t = MLPClassifier(128, self.node_num_t, 64, 2).cuda()
        # self.model_s = MLPClassifier(128, self.node_num_s, 64, 2).cuda()

        # self.model_t = DGCNN_L(self.feat_num, self.node_num_t, 10, 128, 2).cuda()
        # self.model_s = DGCNN_L(self.feat_num, self.node_num_s, 10, 128, 2).cuda()

        self.model_t = DGCNN_encoder_moco(self.feat_num, self.node_num_t, 10, 128, 2).cuda()
        self.model_s = DGCNN_encoder_moco(self.feat_num, self.node_num_s, 10, 32, 2).cuda()

        # self.model_t = DynamicEdgeGNN().cuda()
        # self.model_s = DynamicEdgeGNN().cuda()

        # self.model_t = DGCNN_encoder_moco_former(self.feat_num, self.node_num_t, 10, 128, 2).cuda()
        # self.model_s = DGCNN_encoder_moco_former(self.feat_num, self.node_num_s, 10, 128, 2).cuda()


        # self.models = GCN_all(self.node_num,128,64).cuda()
        # self.models = nn.DataParallel(self.models, device_ids=[i for i in range(len(gpus))])
        # self.model_s = self.model_s.cuda()
        # summary(self.models, (1, 22, 1000))

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
        self.optimizer = torch.optim.Adam(self.model_s.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        node_feat_test_t, node_feat_test_s = Variable(node_feat_test_t.type(self.Tensor)), Variable(node_feat_test_s.type(self.Tensor))
        # edge_matrix_test_t, edge_matrix_test_s = Variable(edge_matrix_test_t.type(self.Tensor)), Variable(edge_matrix_test_s.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        averAUC = 0
        averPre = 0
        averRec = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        loss_lsp_npy = []
        loss_logit_npy = []
        loss_mlp_npy = []

        train_loss_npy, train_acc_npy, test_loss_npy, acc_npy, auc_npy, f1_npy = [], [], [], [], [], []



        # saved_state_dict = torch.load(
        #     '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/exp_results/cl_mae/hbn_cl_mae_t.pth')
        # self.model_t.load_state_dict(saved_state_dict)
        # print('Load Pretrained Teacher Model Successfully')
        #
        # saved_state_dict = torch.load(
        #     '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/exp_results/cl_mae/hbn_cl_mae_s.pth')
        # self.model_s.load_state_dict(saved_state_dict)
        # print('Load Pretrained Student Model Successfully')

        for e in range(self.n_epochs):

            self.model_s.train()
            for i, (node_feat_train_t, node_feat_train_s, train_label) in enumerate(self.dataloader):
                node_feat_train_t, node_feat_train_s = Variable(node_feat_train_t.cuda().type(self.Tensor)), Variable(node_feat_train_s.cuda().type(self.Tensor))
                # edge_matrix_train_t, edge_matrix_train_s = Variable(edge_matrix_train_t.cuda().type(self.Tensor)), Variable(edge_matrix_train_s.cuda().type(self.Tensor))
                train_label = Variable(train_label.cuda().type(self.LongTensor))

                # # data augmentation
                # aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                # img = torch.cat((img, aug_data))
                # label = torch.cat((label, aug_label))

                # outputs_t, emb_t, adj_t, _ = self.model_t(node_feat_train_t)
                # outputs_s, emb_s, adj_s, _ = self.model_s(node_feat_train_s)

                train_edge_index_t, train_edge_attr_t = self.model_t.precompute_edges_and_attrs(node_feat_train_t)
                train_edge_index_s, train_edge_attr_s = self.model_s.precompute_edges_and_attrs(node_feat_train_s)
                outputs_t, emb_t, _, _ = self.model_t(node_feat_train_t, train_edge_index_t, train_edge_attr_t)
                outputs_s, emb_s, _, _ = self.model_s(node_feat_train_s, train_edge_index_s, train_edge_attr_s)

                # soft_loss_value = soft_loss(outputs_t, outputs_s)

                # gnn_lsp_loss = self.lsp_loss(emb_s, adj_s, emb_t, adj_t, self.mask_id)
                # mlp_hop_loss = ncontrast_loss(emb_s, adj_t, self.mask_id, 2, 0.2, threshold=0.2)

                # loss = self.criterion_cls(outputs_s, train_label) + 0.1*soft_loss_value + 0.001*gnn_lsp_loss
                # loss = self.criterion_cls(outputs_s, train_label) + 0.001*gnn_lsp_loss  ##0.001 for linear
                # loss = self.criterion_cls(outputs_s, train_label) + 0.1*mlp_hop_loss
                loss = self.criterion_cls(outputs_s, train_label)


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # test process
            if (e + 1) % 1 == 0:
                self.model_s.eval()
                train_edge_index_s, train_edge_attr_s = self.model_s.precompute_edges_and_attrs(node_feat_train_s)
                Cls, _, _, _ = self.model_s(node_feat_test_s, train_edge_index_s, train_edge_attr_s)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                auc = roc_auc_score(test_label.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

                # 计算混淆矩阵
                cm = confusion_matrix(test_label.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

                # 计算 True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN)
                TP = cm[1, 1]
                FP = cm[0, 1]
                TN = cm[0, 0]
                FN = cm[1, 0]

                # 计算指标
                accuracy = (TP + TN) / (TP + FP + TN + FN)
                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)
                precision = TP / (TP + FP)
                recall = sensitivity
                f1_score = 2 * (precision * recall) / (precision + recall)

                train_pred = torch.max(outputs_s, 1)[1]
                train_acc = float((train_pred == train_label).cpu().numpy().astype(int).sum()) / float(train_label.size(0))

                print('Epoch:', e,
                      '  Train losses: %.4f' % loss.detach().cpu().numpy(),
                      '  Test losses: %.4f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.4f' % train_acc,
                      '  Test AUC is %.4f' % auc,
                      '  Test accuracy is %.4f' % accuracy,
                      # '  soft dis loss is %.4f' % soft_loss_value.detach().cpu().numpy(),
                      # '  LSP loss is %.4f' % gnn_lsp_loss,
                      # '  MLP hop loss is %.4f' % mlp_hop_loss,
                      # '  Test Sensitivity is %.4f' % sensitivity,
                      # '  Test Specificity is %.4f' % specificity,
                      '  Test Precision is %.4f' % precision,
                      '  Test Recall is %.4f' % recall,
                      '  Test F1 is %.4f' % f1_score)

                # self.log_write.write(str(e) + "    " + str(acc) + "\n")

                # loss_lsp_npy.append(gnn_lsp_loss.item())
                # loss_logit_npy.append(soft_loss_value.detach().cpu().numpy())
                # loss_logit_npy.append(soft_loss_value.item())
                # # loss_mlp_npy.append(mlp_hop_loss.item())
                #
                # train_loss_npy.append(loss.detach().cpu().numpy())
                # train_acc_npy.append(train_acc)
                # test_loss_npy.append(loss_test.detach().cpu().numpy())
                # acc_npy.append(accuracy)
                # auc_npy.append(auc)
                # f1_npy.append(f1_score)

                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred

                    torch.save(self.models.state_dict(), os.path.join(self.save_path, 'model_{:.3f}.pth'.format(bestAcc)))
                    torch.save(self.model_t.state_dict(), os.path.join(self.save_path, 'model_abla.pth'.format(bestAcc)))
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        # self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        # self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        # np.save(os.path.join(self.save_path, 'loss_lsp_ebc54_24.npy'), np.array(loss_lsp_npy))
        # # np.save(os.path.join(self.save_path, 'loss_logit.npy'), np.array(loss_logit_npy))
        # # np.save(os.path.join(self.save_path, 'loss_mlp.npy'), np.array(loss_mlp_npy))
        #
        # np.save(os.path.join(self.save_path, 'train_loss.npy'), np.array(train_loss_npy))
        # np.save(os.path.join(self.save_path, 'train_acc.npy'), np.array(train_acc_npy))
        # np.save(os.path.join(self.save_path, 'test_loss.npy'), np.array(test_loss_npy))
        # np.save(os.path.join(self.save_path, 'acc.npy'), np.array(acc_npy))
        # np.save(os.path.join(self.save_path, 'auc.npy'), np.array(auc_npy))
        # np.save(os.path.join(self.save_path, 'f1.npy'), np.array(f1_npy))


        return bestAcc, averAcc, Y_true, Y_pred



