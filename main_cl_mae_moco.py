import argparse
import os

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

from torch.utils.data import DataLoader
from torch.autograd import Variable

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from DGCNN import *
from GCN import *
from tools.HGNN_X2H import *

# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from train_cl_mae_moco import ExP

cudnn.benchmark = False
cudnn.deterministic = True



def main():
    # path_t 和 path_s 路径定义
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_new_allfeat.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_new_24c_allfeat.npy'
    path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_54_pt.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_sex_reo_54_2024.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_54_pt_2000.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd_reo_128_2024.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_54_pt.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd_reo_32_2024.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_sex_reo_31_2024.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_31_pt_2000.npy'
    path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_31_pt.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_12_pt.npy'


    # 创建 ExP 实例
    exp = ExP(1)

    # 获取数据
    total_data_t, total_label_t = exp.get_embarc_graph_data(path_t)
    total_data_s, total_label_s = exp.get_embarc_graph_data(path_s)

    # 直接使用整个数据集进行训练
    print('******************** Training *************************')

    data_t = [total_data_t, total_data_t, total_label_t, total_label_t]
    data_s = [total_data_s, total_data_s, total_label_s, total_label_s]

    # 进行训练
    loss_value = exp.train(data_t, data_s)

    print('Pre-training Done!')

if __name__ == "__main__":
    main()
