# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 11:52:42 2022

@author: Moritz
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

df_toxcast_filtered = pd.read_csv('mowal/Imputation_Paper/Data/Datasets/toxcast_filtered.csv')

train_labels = {}
test_labels = {}

for col in df_toxcast_filtered.columns[:-1]:
    x = df_toxcast_filtered[col].dropna().index
    mapping_indices = {}
    for i,j in enumerate(x):
        mapping_indices[i] = j
    y = df_toxcast_filtered[col].dropna().values
    
    
    #sss takes indices from 0 not the indices in x
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=73)
    for train_index, test_index in sss.split(x, y):
        train_labels[col] = [mapping_indices[i] for i in train_index]
        test_labels[col] = [mapping_indices[i] for i in test_index]
        

#apply splitting given train and test split, convert train labels to nan in test file and vice versa 
df_train = df_toxcast_filtered.copy()
df_test = df_toxcast_filtered.copy()

for col in df_toxcast_filtered.columns[:-1]:
    for tr_label in train_labels[col]:
        df_test.loc[tr_label,col] = np.nan
    for te_label in test_labels[col]:
        df_train.loc[te_label,col] = np.nan
        
df_train.to_csv('mowal/Imputation_Paper/Data/Train_Test_Splits/train_set_assay_based_Toxcast.csv',index=False)
df_test.to_csv('mowal/Imputation_Paper/Data/Train_Test_Splits/test_set_assay_based_Toxcast.csv',index=False)