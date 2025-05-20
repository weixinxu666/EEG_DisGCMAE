import os
import scipy.io as scio
import pandas as pd
import numpy as np
from tqdm import tqdm
from nilearn.connectome import ConnectivityMeasure
from tqdm import tqdm
import re

import warnings

warnings.filterwarnings("ignore")

label_path = '/home/xinxu/Lehigh/Codes/BICLab_data/HBN/BasicDemos_Apr2024.csv'

labels = pd.read_csv(label_path)

# labels = pd.read_csv(label_path, encoding='utf-8')

subject = labels['Identifiers']

lb = labels['Basic_Demos,Sex']

flag_ad = 0
flag_hc = 0

num = len(subject)

all = []
sub = []
phenotype = []

for i, sub_name in enumerate(tqdm(subject)):
    if not np.isnan(lb[i]):
        sex = int(lb[i])


        all.append([sub_name.split(',')[0], sex])
        sub.append(sub_name.split(',')[0])
        phenotype.append(sex)




sub_name_mat = np.array(sub)
labels_adhd_mat = np.array(phenotype)

data_dict = {"subjectID": sub_name_mat, 'labels': labels_adhd_mat}


np.save('./hbn_sex_labels.npy', data_dict)
# np.save('./hbn_adhd_all_labels.npy', data_dict)
# np.save('./hbn_mdd_labels.npy', data_dict)

print('ok')
