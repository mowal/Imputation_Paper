# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:51:42 2021

@author: Moritz
"""

import pandas as pd
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import matthews_corrcoef

def fill_matrix(array): #function that fills Tanimoto matrix: 1 in diagonal; [i,j] = [j,i] and vice versa 
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if i==j:
                array[i,j] = 1
            if math.isnan(array[i,j]):
                array[i,j] = array[j,i]
        
            if math.isnan(array[j,i]):
                array[j,i] = array[i,j]
    return(array)


df_train = pd.read_csv('mowal/Imputation_Paper/Data/Train_Test_Splits/train_set_assay_based_Ames.csv')
assays = df_train.columns[:-1]
dict_indices_train = {}

for assay in assays:
    dict_indices_train[assay] = []
    for i,j in enumerate(df_train[assay]):
        if not math.isnan(j):
            dict_indices_train[assay].append(i)
            
df_test = pd.read_csv('mowal/Imputation_Paper/Data/Train_Test_Splits/test_set_assay_based_Ames.csv')
assays = df_test.columns[:-1]
dict_indices_test = {}

for assay in assays:
    dict_indices_test[assay] = []
    for i,j in enumerate(df_test[assay]):
        if not math.isnan(j):
            dict_indices_test[assay].append(i)



tani = pd.read_csv('tanimoto_similarities_Ames.txt',header=None) # note that the .txt file is not stored in the reporsitory due to its size, can be generated using the compute_Tanimoto_similarity.py script 
tani.columns = ['C1', 'C2', 'Tanimoto']

#add rows for first and last object to get correct format in pivot table 
tani = tani.append({'C1':0,'C2':0,'Tanimoto':np.nan},ignore_index=True)
tani = tani.append({'C1':df_train.shape[0]-1,'C2':df_train.shape[0]-1,'Tanimoto':np.nan},ignore_index=True)
tani = tani.astype({'C1': 'int64'})
tani = tani.astype({'C2': 'int64'})
df_sim = tani.pivot(index='C1', columns='C2', values='Tanimoto')

df_sim.set_index('C1',inplace=True)


#fill matrix
df_sim_filled = fill_matrix(np.array(df_sim))

#transform from similarity to distance
dist_arr = np.full((df_sim_filled.shape[0],df_sim_filled.shape[1]),1) - df_sim_filled


#get 5nn similarities for each assay using sklearn's kNN
clf = KNeighborsClassifier(n_neighbors=5,metric='precomputed')

dict_assay_comp_sims = {}

for assay in assays:
    
    dict_assay_comp_sims[assay] = {}
    
    #get 'train matrix' for assay
    train_dist_assay_cols = dist_arr[:,dict_indices_train[assay]]
    train_dist_assay = train_dist_assay = train_dist_assay_cols[dict_indices_train[assay],:]
    
    #fit knn, dummy y=1
    clf.fit(train_dist_assay,y=[1 for i in range(train_dist_assay.shape[0])])
    
    #get test matrix for assay
    test_dist_assay = np.empty([len(dict_indices_test[assay]),len(dict_indices_train[assay])])
    
    #iterate over test and train indices to fill matrix
    for row,te_idx in enumerate(dict_indices_test[assay]):
        for col,tr_idx in enumerate(dict_indices_train[assay]):
            test_dist_assay[row,col] = dist_arr[te_idx,tr_idx]
    
    #get matrix with 5nn
    nn_matrix = clf.kneighbors(test_dist_assay)[0]
    
    #compute 5nn sims for each assay and compound and add to dict_assay_comp_sims
    for nn_list,orig_idx in zip(nn_matrix,dict_indices_test[assay]):
        dict_assay_comp_sims[assay][orig_idx] = 1 - np.mean(nn_list)
        
        

#bins <0.4, 0.4-0.6, >0.6
#get for each assay test_indices that fall in each bin
dict_assay_bin_indices = {}

for assay in assays:
    dict_assay_bin_indices[assay] = {}
    
    dict_assay_bin_indices[assay]['0.4'] = []
    dict_assay_bin_indices[assay]['0.6'] = []
    dict_assay_bin_indices[assay]['1'] = []
    
    for te_idx in dict_assay_comp_sims[assay]:
        
        if dict_assay_comp_sims[assay][te_idx]<0.4:
            dict_assay_bin_indices[assay]['0.4'].append(te_idx)
        elif dict_assay_comp_sims[assay][te_idx]>0.6:
            dict_assay_bin_indices[assay]['1'].append(te_idx)
        else:
            dict_assay_bin_indices[assay]['0.6'].append(te_idx)
            
            
#get a dictionary to store all true labels according to bin
dict_assay_bin_true = {}
for assay in assays:
    dict_assay_bin_true[assay] = {}
    for bin_ in ['0.4','0.6','1']:
        dict_assay_bin_true[assay][bin_] = list(df_test[assay][dict_assay_bin_indices[assay][bin_]])
        

#import predictions
df_dnn_single = pd.read_csv('mowal/Imputation_Paper/Results/Predictions/predictions_dnn_assay_based_Ames.csv')
df_dnn_fn = pd.read_csv('mowal/Imputation_Paper/Results/Predictions/predictions_dnn_fn_assay_based_Ames.csv')
df_xgb_single = pd.read_csv('mowal/Imputation_Paper/Results/Predictions/predictions_xgb_assay_based_Ames.csv')
df_xgb_fn = pd.read_csv('mowal/Imputation_Paper/Results/Predictions/predictions_xgb_fn_assay_based_Ames.csv')
df_rf_single = pd.read_csv('mowal/Imputation_Paper/Results/Predictions/predictions_rf_assay_based_Ames.csv')
df_rf_fn = pd.read_csv('mowal/Imputation_Paper/Results/Predictions/predictions_rf_fn_assay_based_Ames.csv')
df_dnn_mt = pd.read_csv('mowal/Imputation_Paper/Results/Predictions/predictions_dnn_mt_assay_based_Ames.csv')
df_macau = pd.read_csv('mowal/Imputation_Paper/Results/Predictions/predictions_macau_assay_based_Ames.csv')



#method -> df
df_dict = {'dnn':df_dnn_single,'dnn_fn':df_dnn_fn,'xgb':df_xgb_single, 'xgb_fn':df_xgb_fn, 'rf':df_rf_single, 'rf_fn':df_rf_fn, 'dnn_mt': df_dnn_mt,
           'macau':df_macau}

#put predictions into single dict
#dict structure: method -> assay -> round -> bin -> list of predictions
dict_predictions = {}
for method in ['dnn','dnn_fn','xgb','xgb_fn','rf','rf_fn','dnn_mt','macau']:
    df_method = df_dict[method].copy()
    dict_predictions[method] = {}
    
    for assay in assays:
        df_method_assay = df_method[df_method['assay']==assay]
        dict_predictions[method][assay] = {}
        
        for round_ in range(20):
            df_method_assay_round = df_method_assay[df_method_assay['round']==round_]
            dict_predictions[method][assay][round_] = {}
            
            for bin_ in ['0.4','0.6','1']:
                assay_bin_indices = dict_assay_bin_indices[assay][bin_]
                dict_predictions[method][assay][round_][bin_] = list(df_method_assay_round[df_method_assay_round['test_index'].isin(assay_bin_indices)]['prediction'])
                
                
#store results (mcc scores) in df with cols: method, assay, round, bin 
method_list = []
assay_list = []
round_list = []
bin_list = []
mcc_list = []


for method in ['dnn','dnn_fn','xgb','xgb_fn','rf','rf_fn','dnn_mt','macau']:
    for assay in assays:
        for round_ in range(20):
            for bin_ in ['0.4','0.6','1']:
                method_list.append(method)
                assay_list.append(assay)
                round_list.append(round_)
                bin_list.append(bin_)
                mcc_list.append(matthews_corrcoef(dict_assay_bin_true[assay][bin_],
                                                  dict_predictions[method][assay][round_][bin_]))
df_scores = pd.DataFrame(data={'method':method_list,'assay':assay_list,'round':round_list,'bin':bin_list,'MCC':mcc_list})
df_scores.to_csv('mowal/Imputation_Paper/Results/Scores/mcc_similartiy_bins_Ames.csv',index=False)
