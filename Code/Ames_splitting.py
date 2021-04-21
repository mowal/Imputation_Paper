# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:05:46 2021

@author: Moritz
"""

import pandas as pd
import numpy as np
import math
import random
from sklearn.model_selection import train_test_split


#compound-based splits

df_agg = pd.read_csv('mowal/Imputation_Paper/Data/Datasets/Ames_aggregated.csv',index_col=False)

#split randomly and see what sparsity values are obtained
train,test = train_test_split(df_agg,test_size=0.2,shuffle=True,random_state=23)

for assay in df_agg.iloc[:,:-1].columns:
    count_train = train.shape[0]
    for i in train[assay]:
        if math.isnan(i) == True:
            count_train-=1
    
    count_test = test.shape[0]
    for i in test[assay]:
        if math.isnan(i) == True:
            count_test-=1
    
train.to_csv('mowal/Imputation_Paper/Data/Train_Test_Splits/train_set_compound_based_Ames.csv',index=False)
test.to_csv('mowal/Imputation_Paper/Data/Train_Test_Splits/test_set_compound_based_Ames.csv',index=False)



#assay-based splits
#get for each assay all indices where data is available (0-6167)
assays = df_agg.columns[:-1]
dict_indices = {}

for assay in assays:
    dict_indices[assay] = []
    for i,j in enumerate(df_agg[assay]):
        if not math.isnan(j):
            dict_indices[assay].append(i)
            

#assign randomly 20% of the indices for each assay to the test set
random.seed(a=47)

dict_indices_test = {}
for assay in assays:
    #get a integer k which is the amount of data points assigned to the test set for each assay
    k = round(0.2*len(dict_indices[assay]))
    dict_indices_test[assay] = random.sample(dict_indices[assay], k=k)
    
#create train_df by removing test instances
df_train = df_agg.copy()
for i,row in df_train.iterrows():
    for assay in assays:
        if i in dict_indices_test[assay]:
            df_train.loc[i,assay] = np.nan
            
#create test_df by retaining test instances
df_test = df_agg.copy()
for i,row in df_test.iterrows():
    for assay in assays:
        if i not in dict_indices_test[assay]:
            df_test.loc[i,assay] = np.nan
            
df_train.to_csv('mowal/Imputation_Paper/Data/Train_Test_Splits/train_set_assay_based_Ames.csv',index=False)
df_test.to_csv('mowal/Imputation_Paper/Data/Train_Test_Splits/test_set_assay_based_Ames.csv',index=False)
