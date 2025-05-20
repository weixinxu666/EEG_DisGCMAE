import numpy as np
import os
from tqdm import tqdm
import glob
import re
from scipy.io import loadmat
import torch
from sklearn.model_selection import train_test_split
from scipy.io import savemat

from scipy.signal import welch
from scipy.signal import welch, butter, filtfilt, lfilter



def get_correlation_matrices(time_series, threshold):
    correlation_matrices = np.corrcoef(time_series.T) #T将数组进行转置，转成(n_rois, n_timepoints)
    # adj_matrix = np.triu(adj_matrix)
    # adj_matrix = np.expand_dims(adj_matrix, axis=0)
    return correlation_matrices  # 100x100


def get_sparse_correlation_matrices(correlation_matrices, threshold):
    adj_matrix = np.zeros_like(correlation_matrices)
    adj_matrix[correlation_matrices >= threshold] = 1
    # adj_matrix = np.triu(adj_matrix)
    # adj_matrix = np.expand_dims(adj_matrix, axis=0)
    return adj_matrix  # 100x100


def unify_time_roi_matrices(time_series_roi, thres):
    time_size = time_series_roi.shape[0]
    if time_size <= thres:
        new_matrix = np.zeros((thres, thres))
        new_matrix[:time_size, :] = time_series_roi
        return new_matrix
    else:
        new_matrix = time_series_roi[:thres, :]
        return new_matrix



def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def cal_eeg_psd_conn(eeg_data, feat_num):
    freq_bands = {
        # 'delta': (0.5, 4),
        #           'theta': (4, 8),
                  'alpha': (8, 13)
        #           'beta': (13, 30),
                  # 'gamma': (30, 100)
    }

    fs = 256  # Sampling frequency
    all_band_psd_matrix = []
    all_band_node_feature = []

    for band, freq_range in freq_bands.items():
        # Filter EEG data for the current frequency band
        filtered_data = np.array([bandpass_filter(channel, freq_range[0], freq_range[1], fs) for channel in eeg_data])

        # Calculate PSD for each channel
        psd_matrix = []
        node_feature = []
        for channel_data in filtered_data:
            freqs, psd = welch(channel_data, fs=fs)
            psd_matrix.append(psd[:feat_num])
            node_feature.append(psd[:feat_num])

        psd_matrix = np.array(psd_matrix)  # Shape: (num_channels, feat_num)

        all_band_psd_matrix.append(psd_matrix)
        all_band_node_feature.append(node_feature)

        # Calculate connectivity matrix (using Pearson correlation coefficient)
        connectivity_matrix = np.corrcoef(psd_matrix)

        # You can return or save the connectivity_matrix for each frequency band if needed

    return all_band_psd_matrix, all_band_node_feature



def filter_eeg_band(data, band, fs):
    nyquist_freq = fs / 2
    low = band[0] / nyquist_freq
    high = band[1] / nyquist_freq
    b, a = butter(4, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=1)
    return filtered_data



def spectral_coherence_matrix(eeg_data):
    num_channels, signal_length = eeg_data.shape
    coherence_matrix = np.zeros((num_channels, num_channels))

    for i in range(num_channels):
        for j in range(i + 1, num_channels):  # 只计算上三角部分，因为是对称矩阵
            # 获取第 i 和第 j 个通道的信号
            signal_i = eeg_data[i]
            signal_j = eeg_data[j]

            # 计算信号的傅立叶变换
            spectrum_i = np.fft.fft(signal_i)
            spectrum_j = np.fft.fft(signal_j)

            # 计算交叉功率谱密度
            cross_power_spectrum = np.abs(spectrum_i * np.conj(spectrum_j)) ** 2

            # 计算功率谱密度
            power_spectrum_i = np.abs(spectrum_i) ** 2
            power_spectrum_j = np.abs(spectrum_j) ** 2

            # 计算谱相干度
            coherence = cross_power_spectrum / (power_spectrum_i * power_spectrum_j)

            # 将计算结果放入对称矩阵中
            coherence_matrix[i, j] = coherence.mean()

    return coherence_matrix + coherence_matrix.T  # 对称矩阵


def compute_spectral_features(eeg_data, fs=10):
    num_channels, num_samples = eeg_data.shape
    spectral_features = []

    # 定义频带范围（示例中为常见的 alpha 和 beta 频带）
    freq_bands = {'alpha': (8, 13), 'beta': (13, 30)}

    for i in range(num_channels):
        channel_features = []
        for band_name, (f_low, f_high) in freq_bands.items():
            # 使用 welch 方法计算功率谱密度
            freqs, psd = welch(eeg_data[i], fs, nperseg=fs*2)
            # 提取特定频带上的功率
            band_power = np.mean(psd[(freqs >= f_low) & (freqs <= f_high)])
            channel_features.append(band_power)
        spectral_features.append(channel_features)
        connectivity_matrix = np.corrcoef(spectral_features)

    return np.array(connectivity_matrix)

def parse_mat(mat_path):
    mat_data = loadmat(mat_path)\

    # EEG array:  [54, tps]  54 channels x time points  (ndarray)
    eeg_data = mat_data['EEG']['data'][0][0]
    return eeg_data


def eeg_downsample(original_matrix, rate):

    # 下采样为 [100, 100] 的矩阵
    downsampled_matrix = original_matrix[:, ::rate]  # 每隔250列取一列

    return downsampled_matrix


def unify_time_points(time_series_roi, thres):
    time_size = time_series_roi.shape[1]
    if time_size <= thres:

        return 'None'
    else:
        new_matrix = time_series_roi[:, :thres]
        return new_matrix


# def unify_time_points(time_series_roi, thres):
#     time_size = time_series_roi.shape[1]
#     if time_size <= thres:
#         new_matrix = np.zeros((100, thres))
#         new_matrix[:, :time_size] = time_series_roi
#         return new_matrix
#     else:
#         new_matrix = time_series_roi[:, :thres]
#         return new_matrix



file_dir = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/EEG_ChannelSignals/HBN_EEG_ChannelSignals_REO'
# file_dir = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/EEG_ChannelSignals/HBN_EEG_ChannelSignals_REC'

subdir_list = os.listdir(file_dir)

name_list = []
eeg_list = []
reo_dir_list = []
eeg_conn_list = []
eeg_node_feat_list = []

for dir in tqdm(subdir_list):
    sub_id, EEG_type = dir.split('_')[0], dir.split('_')[3]

    if EEG_type == 'REO':
    # if EEG_type == 'REC':

        reo_mat_path = os.path.join(file_dir, dir)
        reo_dir_list.append(reo_mat_path)
        eeg_data = parse_mat(reo_mat_path)

        ## removed electrodes



        # print('name: ', dir, 'shape: ', eeg_data.shape)
        # print(eeg_data.shape)

        if eeg_data.shape[1] >= 25000:
            eeg_unified = unify_time_points(eeg_data, thres=25000)
            eeg_ds = eeg_downsample(eeg_unified, rate=1)

            # selected_channels = [30, 31, 17, 18, 4, 5, 38, 39]    # 8 channels
            # selected_channels = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127]
            # selected_channels = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125]
            # eeg_ds = eeg_ds[selected_channels, :]

            name_list.append(sub_id)
            eeg_list.append(eeg_ds)
            eeg_node_feat, eeg_conn = cal_eeg_psd_conn(eeg_ds, 128)
            # eeg_conn = compute_spectral_features(eeg_ds)
            # eeg_conn = spectral_coherence_matrix(eeg_ds[:,:1000])
            eeg_conn_list.append(eeg_conn)
            eeg_node_feat_list.append(eeg_node_feat)


# embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_neuro_labels.npy'
# embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_axt_labels.npy'
# embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_sex_labels.npy'
# embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_asd_labels.npy'
# embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_adhd_labels.npy'
# embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_mdd_labels.npy'
# embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_mdd_labels_2024.npy'
embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_asd_labels_2024.npy'
# embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_axt_labels_2024.npy'
# embarc_sex_labels = '/home/xinxu/Lehigh/Codes/lehigh_eeg/EEG-Conformer-main/Data/HBN/hbn_neuro_labels_2024.npy'
labels_sex = np.load(embarc_sex_labels, allow_pickle=True).item()



labels_id_list = labels_sex['subjectID']
labels_sex_list = labels_sex['labels']

dict_eeg_ts_id = {key: value for key, value in zip(name_list, eeg_list)}
dict_eeg_conn_id = {key: value for key, value in zip(name_list, eeg_conn_list)}
dict_eeg_node_feat_id = {key: value for key, value in zip(name_list, eeg_node_feat_list)}
dict_sex_id = {key: value for key, value in zip(labels_id_list.tolist(), labels_sex_list.tolist())}


data_ts = []
data_node = []
data = []
labels = []



for id in dict_eeg_conn_id.keys():
    try:
        print(id)
        eeg_ts = dict_eeg_ts_id[id]
        eeg_node_feat = dict_eeg_node_feat_id[id]
        eeg_conn = dict_eeg_conn_id[id]
        label = dict_sex_id[id]

        data_ts.append(eeg_ts)
        data_node.append(eeg_node_feat)
        data.append(eeg_conn)
        labels.append(label)
    except KeyError:
        print('No ID', id)


data_ts = np.array(data_ts)
data_node = np.array(data_node)
data = np.array(data)

print(labels.count(1))
labels = np.array(labels)

#Permutation tasks
# labels = np.random.randint(2, size=(308,), dtype=np.int64)

# data_sparse = get_correlation_matrices(data, threshold=0.3)

data_dict = {"eeg_ts": data_node, "eeg_corr": data, "labels": labels}
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_asd.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_neuro.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_axt.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_sex.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_adhd.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd_rec_64_2024.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd_reo_64_2024.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd_reo_64.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd_reo_32.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd_reo_128.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd_reo_128_band_al.npy', data_dict)
np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_asd_reo_128_band_al.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_asd_reo_128_2024.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_axt_reo_128_2024.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_neuro_reo_128_2024.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_mdd_reo_64_2024.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_asd_reo_128.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_asd_reo_64.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_asd_reo_32.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_asd_rec_64.npy', data_dict)
# np.save('/home/xinxu/Lehigh/Codes/BICLab_data/HBN/hbn_channel_eeg_ts_conn_asd_rec_128.npy', data_dict)


# data_dict_ts = {"samples": data_ts, "labels": labels}
# np.save('/home/xinxu/Lehigh/Codes/Lehigh_graph/xxw_lehigh/data/EMBARC/embarc_channel_eeg_ts_hamd.npy', data_dict_ts)


# train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, shuffle=True, random_state=42)
# train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.1, shuffle=True,
#                                                                   random_state=42)
#
# # 输出划分后的数据集大小
# print("训练集大小:", len(train_data))
# print("验证集大小:", len(val_data))
# print("测试集大小:", len(test_data))
#
# train_dict = {'samples': torch.tensor(train_data).double(), 'labels': torch.tensor(train_labels)}
# val_dict = {'samples': torch.tensor(val_data).double(), 'labels': torch.tensor(val_labels)}
# test_dict = {'samples': torch.tensor(test_data).double(), 'labels': torch.tensor(test_labels)}

# train_dict = {'samples': np.array(train_data), 'labels': np.array(train_labels)}
# val_dict = {'samples': np.array(val_data), 'labels': np.array(val_labels)}
# test_dict = {'samples': np.array(test_data), 'labels': np.array(test_labels)}

# 保存为 .pt 文件
# torch.save(train_dict, '/home/xinxu/Lehigh/Codes/Lehigh_graph/TFC-pretraining-main/datasets/EMBARC_channel/train.pt')
# torch.save(val_dict, '/home/xinxu/Lehigh/Codes/Lehigh_graph/TFC-pretraining-main/datasets/EMBARC_channel/val.pt')
# torch.save(test_dict, '/home/xinxu/Lehigh/Codes/Lehigh_graph/TFC-pretraining-main/datasets/EMBARC_channel/test.pt')

# torch.save(train_dict, '/home/xinxu/Lehigh/Codes/Lehigh_graph/TFC-pretraining-main/datasets/EMBARC_few_channel/train.pt')
# torch.save(val_dict, '/home/xinxu/Lehigh/Codes/Lehigh_graph/TFC-pretraining-main/datasets/EMBARC_few_channel/val.pt')
# torch.save(test_dict, '/home/xinxu/Lehigh/Codes/Lehigh_graph/TFC-pretraining-main/datasets/EMBARC_few_channel/test.pt')

# savemat('train.mat', train_dict)
# savemat('val.mat', val_dict)
# savemat('test.mat', test_dict)


print('ok')
