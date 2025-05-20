# 完整电极列表
full_eeg_channels = [
    'AF3', 'AF4', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz',
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FCz',
    'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4',
    'PO7', 'PO8', 'POz', 'Pz', 'T7', 'T8', 'TP7', 'TP8'
]

# 32通道EEG系统电极标签
eeg_channels_32 = ['AF3', 'AF4', 'C3', 'C4', 'CP1', 'CP2', 'CP5', 'CP6',
                    'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'Fp1',
                    'Fp2', 'Fz', 'O1', 'O2', 'Oz', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'Pz', 'T7', 'T8']

# 16通道EEG系统电极标签
eeg_channels_16 = ['C3', 'C4', 'F3', 'F4', 'Fp1', 'Fp2', 'O1', 'O2', 'P3', 'P4', 'T7', 'T8']

# 查找32通道和16通道电极在完整列表中的索引
indices_32 = [full_eeg_channels.index(channel) for channel in eeg_channels_32 if channel in full_eeg_channels]
indices_16 = [full_eeg_channels.index(channel) for channel in eeg_channels_16 if channel in full_eeg_channels]

print("32通道EEG电极在完整列表中的索引:")
print(indices_32)

print("\n16通道EEG电极在完整列表中的索引:")
print(indices_16)
