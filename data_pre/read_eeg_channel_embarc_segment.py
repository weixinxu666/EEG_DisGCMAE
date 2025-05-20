import numpy as np
import os
from tqdm import tqdm
import re
from scipy.io import loadmat
from scipy.signal import welch
import torch

def get_correlation_matrices(time_series, threshold):
    correlation_matrices = np.corrcoef(time_series.T)  # T将数组进行转置，转成(n_rois, n_timepoints)
    return correlation_matrices  # 100x100


def get_sparse_correlation_matrices(correlation_matrices, threshold):
    adj_matrix = np.zeros_like(correlation_matrices)
    adj_matrix[correlation_matrices >= threshold] = 1
    return adj_matrix  # 100x100


def cal_eeg_psd_conn(eeg_data, feat_num):
    num_segments = eeg_data.shape[1] // 2000  # 计算可以分成多少个1000时间点的段
    psd_matrices = []
    node_features = []
    connectivity_matrices = []

    for i in range(num_segments):
        segment_data = eeg_data[:, i * 2000:(i + 1) * 2000]  # 切分时间序列
        psd_matrix = []
        for channel_data in segment_data:
            freqs, psd = welch(channel_data, fs=256)
            psd_matrix.append(psd[:feat_num])
        psd_matrix = np.array(psd_matrix)  # 转换为 numpy 数组，形状为 (54, num_freq_bins)
        psd_matrices.append(psd_matrix)
        connectivity_matrix = np.corrcoef(psd_matrix)
        connectivity_matrices.append(connectivity_matrix)
        node_features.append(psd_matrix)

    return connectivity_matrices, node_features


def parse_mat(mat_path):
    mat_data = loadmat(mat_path)
    eeg_data = mat_data['EEG']['data'][0][0]
    return eeg_data


def eeg_downsample(original_matrix, rate):
    downsampled_matrix = original_matrix[:, ::rate]  # 每隔250列取一列
    return downsampled_matrix


def unify_time_points(time_series_roi, thres):
    time_size = time_series_roi.shape[1]
    if time_size <= thres:
        new_matrix = np.zeros((100, thres))
        new_matrix[:, :time_size] = time_series_roi
        return new_matrix
    else:
        new_matrix = time_series_roi[:, :thres]
        return new_matrix


file_dir = '/home/xinxu/Lehigh/Codes/Lehigh_graph/TFC-pretraining-main/datasets/EEG_ChannelSignals_EMBARC_Baseline'

subdir_list = os.listdir(file_dir)

name_list = []
eeg_list = []
eeg_conn_list = []
eeg_node_feat_list = []

for dir in tqdm(subdir_list):
    sub_id, EEG_type = dir.split('_')[1], dir.split('_')[2]

    # if EEG_type == 'REO':
    mat_file = os.listdir(os.path.join(file_dir, dir))
    pattern = re.compile(r'.*line\.mat')

    matching_files = [file_name for file_name in mat_file if re.match(pattern, file_name)]
    reo_mat_path = os.path.join(file_dir, dir, matching_files[0])
    eeg_data = parse_mat(reo_mat_path)

    eeg_unified = unify_time_points(eeg_data, thres=22000)
    eeg_ds = eeg_downsample(eeg_unified, rate=1)

    # selected_channels = [30, 31, 17, 18, 4, 5, 38, 39]  # 8 channels
    # selected_channels = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]
    # 32sys
    # selected_channels = [33, 34, 32, 17, 18, 15, 16, 24, 25, 28, 29, 10, 11, 43, 44, 8, 9, 12, 13, 38, 39, 41, 42, 43, 46, 47, 40, 31, 32, 41, 37, 38]
    #16 sys
    # selected_channels = [33, 34, 32, 17, 18, 15, 16, 24, 25, 10, 11, 43, 44, 38, 39, 40]
    # 31  sys
    # selected_channels = [0, 1, 4, 5, 8, 9, 12, 13, 17, 18, 21, 22, 23, 24, 27, 28, 30, 31, 32, 33, 34, 35, 38, 39, 42, 43, 44, 45, 49, 50, 51]
    #12  sys
    selected_channels = [4, 5, 17, 18, 30, 31, 33, 34, 38, 39, 50, 51]

    eeg_ds = eeg_ds[selected_channels, :]

    # 切分时间序列并计算每段的PSD
    connectivity_matrices, node_features = cal_eeg_psd_conn(eeg_ds, 128)

    for i, connectivity_matrix in enumerate(connectivity_matrices):
        eeg_conn_list.append(connectivity_matrix)
        eeg_node_feat_list.append(node_features[i])
        name_list.append(sub_id)  # 每个小段使用相同的sub_id

# Load labels
embarc_sex_labels = '/home/xinxu/Lehigh/Codes/Lehigh_graph/TFC-pretraining-main/datasets/embarc_labels_sex.npy'
# embarc_sex_labels = '/home/xinxu/Lehigh/Codes/Lehigh_graph/TFC-pretraining-main/datasets/embarc_labels_hamd.npy'
labels_sex = np.load(embarc_sex_labels, allow_pickle=True).item()

labels_id_list = labels_sex['subjectID']
labels_sex_list = labels_sex['labels']

# 创建一个字典来映射sub_id到性别标签
id_to_label = {id_: label for id_, label in zip(labels_id_list, labels_sex_list)}

data_node = []
data = []
labels = []

for id in name_list:
    if id in id_to_label:
        data_node.append(eeg_node_feat_list[name_list.index(id)])
        data.append(eeg_conn_list[name_list.index(id)])
        labels.append(id_to_label[id])

data_node = np.array(data_node)
data = np.array(data)
labels = np.array(labels)

data_dict = {"eeg_ts": data_node, "eeg_corr": data, "labels": labels}
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_54_pt_2000.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_31_pt_2000.npy', data_dict)
np.save('/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_12_pt_2000.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_31_pt.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_12_pt.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/EMBARC/embarc_channel_eeg_ts_conn_reo_24_pt.npy', data_dict)

print('ok')
