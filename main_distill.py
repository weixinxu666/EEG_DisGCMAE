import argparse
import os

gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))


from torch.utils.data import DataLoader
from torch.autograd import Variable

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, roc_auc_score

from DGCNN import *
from GCN import *
from tools.HGNN_X2H import *

# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from train_distill import ExP

cudnn.benchmark = False
cudnn.deterministic = True




def main():

    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_new_allfeat.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_new_8c_allfeat.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_new_24c_allfeat.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd_reo_64.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd_reo_128.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd_reo_128_2024.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_sex_reo_128.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_asd_reo_128_2024.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_axt_reo_128_2024.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_neuro_reo_128_2024.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_new_sex_rec.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_sex_reo_54.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_new_hamd_rec.npy'
    path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_hamd_reo_54.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_new_24c_allfeat.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_hamd_reo_54.npy'
    # path_t = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_new_8c_allfeat.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_new_allfeat.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_new_24c_allfeat.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_new_8c_allfeat.npy'
    path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_hamd_reo_54.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd_reo_64.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd_reo_128_2024.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd_reo_32_2024.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd_reo_32_2024.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_sex_reo_128.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_asd_reo_128_2024.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_axt_reo_128_2024.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_neuro_reo_128_2024.npy'
    # path_s = '/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_sex_reo_54.npy'


    avg_acc = []
    avg_auc = []
    avg_f1 = []
    avg_precision = []
    avg_recall = []
    avg_sensitivity = []
    avg_specificity = []

    all_true_labels = []
    all_predictions = []

    exp = ExP(1)
    total_data_t, total_label_t = exp.get_embarc_graph_data(path_t)
    total_data_s, total_label_s = exp.get_embarc_graph_data(path_s)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10折交叉验证的划分器

    fold = 0
    for train_index, test_index in kf.split(total_data_t):
        fold += 1

        train_data_t, test_data_t = total_data_t[train_index], total_data_t[test_index]
        train_label_t, test_label_t = total_label_t[train_index], total_label_t[test_index]
        train_ts_t = train_data_t[:,:,:]
        test_ts_t = test_data_t[:,:,:]

        train_data_s, test_data_s = total_data_s[train_index], total_data_s[test_index]
        train_label_s, test_label_s = total_label_s[train_index], total_label_s[test_index]
        train_ts_s = train_data_s[:,:,:]
        test_ts_s = test_data_s[:,:,:]

        print('******************** Training Fold %d *************************' % fold)


        exp = ExP(fold)

        data_t = [train_ts_t, test_ts_t, train_label_t, test_label_t]
        data_s = [train_ts_s, test_ts_s, train_label_s, test_label_s]

        bestAcc, averAcc, Y_true, Y_pred = exp.train(data_t, data_s)

        Y_true_cpu = Y_true.cpu().numpy()
        Y_pred_cpu = Y_pred.cpu().numpy()

        acc = accuracy_score(Y_true_cpu, Y_pred_cpu)
        auc = roc_auc_score(Y_true_cpu, Y_pred_cpu)
        f1 = f1_score(Y_true_cpu, Y_pred_cpu)
        precision = precision_score(Y_true_cpu, Y_pred_cpu)
        recall = recall_score(Y_true_cpu, Y_pred_cpu)

        cm = confusion_matrix(Y_true_cpu, Y_pred_cpu)
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TP = cm[1, 1]
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        all_true_labels.extend(Y_true_cpu)
        all_predictions.extend(Y_pred_cpu)

        avg_acc.append(acc)
        avg_auc.append(auc)
        avg_f1.append(f1)
        avg_precision.append(precision)
        avg_recall.append(recall)
        avg_sensitivity.append(sensitivity)
        avg_specificity.append(specificity)

    avg_acc = np.mean(avg_acc)
    avg_auc = np.mean(avg_auc)
    avg_f1 = np.mean(avg_f1)
    avg_precision = np.mean(avg_precision)
    avg_recall = np.mean(avg_recall)
    avg_sensitivity = np.mean(avg_sensitivity)
    avg_specificity = np.mean(avg_specificity)

    print('\n\n')

    print('Average accuracy across all folds:', avg_acc)
    print('Average AUC across all folds:', avg_auc)
    print('Average F1 score across all folds:', avg_f1)
    print('Average precision across all folds:', avg_precision)
    print('Average recall across all folds:', avg_recall)
    print('Average sensitivity across all folds:', avg_sensitivity)
    print('Average specificity across all folds:', avg_specificity)



    # 绘制混淆矩阵
    cm = confusion_matrix(all_true_labels, all_predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))

    # 绘制热图，设置 annot_kws 参数来调整字体大小
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", annot_kws={"size": 14})

    # 设置坐标轴标签和标题的字体大小
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    # plt.title('HBN MDD Classification', fontsize=18)
    plt.title('EMBARC Severity Grading', fontsize=18)
    # plt.savefig('cm_hbn_mdd2.pdf')
    # plt.savefig('cm_emb_sex.pdf')
    plt.savefig('cm_emb_hamd.pdf')
    plt.show()


if __name__ == "__main__":
    main()