
import argparse
import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import datetime


from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix

# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# writer = SummaryWriter('./TensorBoardX/')


# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()


        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (54, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(1080, 256),
            # nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=2, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class ExP():
    def __init__(self, nsub):
        super(ExP, self).__init__()
        self.batch_size = 72
        self.n_epochs = 50
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.nSub = nsub

        self.start_epoch = 0
        self.root = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/EMBARC/'

        self.log_write = open("./results/log_subject%d.txt" % self.nSub, "w")


        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = Conformer().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        # summary(self.models, (1, 22, 1000))



    def get_embarc_source_data(self):
        # train data
        path = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/EMBARC/embarc_channel_eeg_ts.npy'
        # path = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/EMBARC/embarc_channel_eeg_ts_hamd.npy'
        self.data =  np.load(path, allow_pickle=True).item()

        self.total_data = self.data['samples'][:,:54,:500]
        self.total_label = self.data['labels']

        self.total_data = np.expand_dims(self.total_data, axis=1)
        self.total_label = np.transpose(self.total_label)

        self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(self.total_data, self.total_label, shuffle=True, test_size=0.2,
                                                                            random_state=42)

        # scaler = StandardScaler()
        # self.train_data = scaler.fit_transform(self.train_data)
        # self.test_data = scaler.transform(self.test_data)

        # standardize
        target_mean = np.mean(self.train_data)
        target_std = np.std(self.test_data)
        self.train_data = (self.train_data - target_mean) / target_std
        self.test_data = (self.test_data - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.train_data, self.train_label, self.test_data, self.test_label



    def train(self):

        img, label, test_data, test_label = self.get_embarc_source_data()

        label = label.squeeze()
        test_label = test_label.squeeze()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        label = torch.tensor(label, dtype=torch.int64)
        test_label = torch.tensor(test_label, dtype=torch.int64)

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the cnn models
        total_step = len(self.dataloader)
        curr_lr = self.lr

        for e in range(self.n_epochs):
            # in_epoch = time.time()
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))


                tok, outputs = self.model(img)

                loss = self.criterion_cls(outputs, label) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            # out_epoch = time.time()


            # test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data)


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

                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                print('Epoch:', e,
                      '  Train losses: %.6f' % loss.detach().cpu().numpy(),
                      '  Test losses: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test AUC is %.6f' % auc,
                      '  Test accuracy is %.6f' % accuracy,
                      '  Test Sensitivity is %.6f' % sensitivity,
                      '  Test Specificity is %.6f' % specificity,
                      '  Test Precision is %.6f' % precision,
                      '  Test Recall is %.6f' % recall,
                      '  Test F1 is %.6f' % f1_score)

                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred

        print('Model saved: Best Acc is:', bestAcc)
        torch.save(self.model.module.state_dict(), 'model.pth')
        # torch.save(self.models.module.state_dict(), 'models.pth')
        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred
        # writer.close()


def main():
    best = 0
    aver = 0
    result_write = open("./results/sub_result.txt", "w")

    for i in range(9):
        starttime = datetime.datetime.now()


        seed_n = np.random.randint(2021)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)


        print('Subject %d' % (i+1))
        exp = ExP(i + 1)

        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))
        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))


    best = best / 9
    aver = aver / 9

    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()


if __name__ == "__main__":

    main()

