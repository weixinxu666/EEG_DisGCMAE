import numpy as np

# 电极坐标
electrode_coordinates = {
    'Fp1': (-0.7, 5.2, 2.5),
    'Fp2': (0.7, 5.2, 2.5),
    'F3': (-2.5, 4.8, 2.5),
    'F4': (2.5, 4.8, 2.5),
    'C3': (-3.5, 3.5, 2.5),
    'C4': (3.5, 3.5, 2.5),
    'P3': (-4.5, 1.5, 2.5),
    'P4': (4.5, 1.5, 2.5),
    'O1': (-2.5, -4.8, 2.5),
    'O2': (2.5, -4.8, 2.5),
    'F7': (-4.2, 6.5, 2.5),
    'F8': (4.2, 6.5, 2.5),
    'T7': (-6.5, 0.0, 2.5),
    'T8': (6.5, 0.0, 2.5),
    'P7': (-7.0, -3.5, 2.5),
    'P8': (7.0, -3.5, 2.5),
    'AF3': (-1.8, 5.0, 3.8),
    'AF4': (1.8, 5.0, 3.8),
    'FC3': (-3.0, 4.0, 3.8),
    'FC4': (3.0, 4.0, 3.8),
    'CP3': (-4.0, 2.0, 3.8),
    'CP4': (4.0, 2.0, 3.8),
    'PO3': (-3.5, -4.0, 3.8),
    'PO4': (3.5, -4.0, 3.8),
    'FT7': (-5.8, 6.0, 3.8),
    'FT8': (5.8, 6.0, 3.8),
    'TP7': (-7.5, 0.0, 3.8),
    'TP8': (7.5, 0.0, 3.8),
    'PO7': (-7.0, -4.0, 3.8),
    'PO8': (7.0, -4.0, 3.8),
    'Fpz': (0.0, 5.5, 3.8),
    'CPz': (0.0, 3.0, 3.8),
    'POz': (0.0, -4.2, 3.8),
    'FCz': (0.0, 4.5, 5.0),
    'Cz': (0.0, 0.0, 5.0),
    'Oz': (0.0, -5.0, 5.0),
    'P9': (-5.0, -2.0, 0.0),
    'P10': (5.0, -2.0, 0.0),
    'PO9': (-6.0, -4.0, 0.0),
    'PO10': (6.0, -4.0, 0.0),
    'TP9': (-8.0, 0.0, 0.0),
    'TP10': (8.0, 0.0, 0.0),
    'Iz': (0.0, 7.0, 0.0),
    'O9': (-5.0, -7.0, 0.0),
    'O10': (5.0, -7.0, 0.0),
    'AFz': (0.0, 4.5, 0.0),
    'AF7': (-5.0, 6.5, 0.0),
    'AF8': (5.0, 6.5, 0.0),
    'FT9': (-7.0, 6.0, 0.0),
    'FT10': (7.0, 6.0, 0.0),
    'P9_1': (-6.0, -1.0, 0.0),
    'P10_1': (6.0, -1.0, 0.0),
    'P9_2': (-5.0, -5.0, 0.0),
    'P10_2': (5.0, -5.0, 0.0)
}

# 计算连接矩阵
def compute_connectivity_matrix(electrode_coordinates):
    n_electrodes = len(electrode_coordinates)
    connectivity_matrix = np.zeros((n_electrodes, n_electrodes))
    for i, (label_i, coord_i) in enumerate(electrode_coordinates.items()):
        for j, (label_j, coord_j) in enumerate(electrode_coordinates.items()):
            # 计算单位向量
            ui = np.array(coord_i) / np.linalg.norm(np.array(coord_i))
            uj = np.array(coord_j) / np.linalg.norm(np.array(coord_j))
            # 计算点积
            dot_product = np.dot(ui, uj)
            # 标准化连接值并存储在连接矩阵中
            connectivity_matrix[i, j] = dot_product / (np.linalg.norm(np.array(coord_i)) ** 2)
    return connectivity_matrix

# 调用函数计算连接矩阵
connectivity_matrix = compute_connectivity_matrix(electrode_coordinates)

# 打印连接矩阵
print(connectivity_matrix)
print(connectivity_matrix.shape)
