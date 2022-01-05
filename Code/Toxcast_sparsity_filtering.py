# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:06:31 2022

@author: Moritz
"""

import pandas as pd
import numpy as np
import random

df_train = pd.read_csv('mowal/Imputation_Paper/Data/Train_Test_Splits/train_set_assay_based_Toxcast.csv')

#keep only 1000 of each assay (randomly sampled) in train set
def train_set_sampling(n_samples,random_seed=36):
    df_train_sampled = df_train.copy()
    random.seed(a=random_seed)
    for icol in range(0,(df_train.shape[1]-1)):

        #get number of available data points
        data_points_assay = df_train_sampled.shape[0]-df_train_sampled.iloc[:,icol].isna().sum()

        #sample to keep n_samples if more than n_samples available
        if data_points_assay >n_samples:
            #get available labels
            av_index = []
            for i,av in enumerate(df_train_sampled.iloc[:,icol].isna()):
                if av == False:
                    av_index.append(i)

            labels_to_keep = random.sample(av_index,k=n_samples)
            labels_to_remove = set(av_index).difference(set(labels_to_keep))
            for i in labels_to_remove:
                df_train_sampled.iloc[i,icol] = np.nan
                
    return df_train_sampled


df_train_sampled = train_set_sampling(1000)
df_train_sampled.to_csv('mowal/Imputation_Paper/Data/Train_Test_Splits/train_set_assay_based_sparse_Toxcast.csv',index=False)
