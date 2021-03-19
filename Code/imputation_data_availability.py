# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:33:31 2021

@author: Moritz
"""

import pandas as pd
from sklearn.metrics import matthews_corrcoef
import math


#get indices for train and test compounds per assay
df_train = pd.read_csv('mowal/Imputation_Paper/Data/train_set_asssay_based_Ames.csv')
assays = df_train.columns[:-1]
dict_indices_train = {}

for assay in assays:
    dict_indices_train[assay] = []
    for i,j in enumerate(df_train[assay]):
        if not math.isnan(j):
            dict_indices_train[assay].append(i)
            
df_test = pd.read_csv('mowal/Imputation_Paper/Data/test_set_assay_based_Ames.csv')
assays = df_test.columns[:-1]
dict_indices_test = {}

for assay in assays:
    dict_indices_test[assay] = []
    for i,j in enumerate(df_test[assay]):
        if not math.isnan(j):
            dict_indices_test[assay].append(i)
            


#get for each compound with at least one label in the test set the number of labels in the train set and store in a dict
dict_number_train_instances = {}

for i,row in df_test.iloc[:,:-1].iterrows():
    #check if index has at least 1 test instance
    has_test = False
    for j in row:
        if not math.isnan(j):
            has_test = True
    
    #get number of train_instances
    number_train = 0
    for k in df_train.iloc[i,:-1]:
        if not math.isnan(k):
            number_train+=1
    dict_number_train_instances[i] = number_train
    


#bins: 0-1,2-3,>3
#get for each assay test_indices that fall in each bin
dict_numbers_assay_bin_indices = {}

for assay in assays:
    dict_numbers_assay_bin_indices[assay] = {}
    
    dict_numbers_assay_bin_indices[assay]['0-1'] = []
    dict_numbers_assay_bin_indices[assay]['2-3'] = []
    dict_numbers_assay_bin_indices[assay]['>3'] = []
    
    for te_idx in dict_indices_test[assay]:
        
        if dict_number_train_instances[te_idx]<2:
            dict_numbers_assay_bin_indices[assay]['0-1'].append(te_idx)
        elif dict_number_train_instances[te_idx]>3:
            dict_numbers_assay_bin_indices[assay]['>3'].append(te_idx)
        else:
            dict_numbers_assay_bin_indices[assay]['2-3'].append(te_idx)
            
            
#get a dictionary to store all true labels according to bin
dict_numbers_assay_bin_true = {}
for assay in assays:
    dict_numbers_assay_bin_true[assay] = {}
    for bin_ in ['0-1','2-3','>3']:
        dict_numbers_assay_bin_true[assay][bin_] = list(df_test[assay][dict_numbers_assay_bin_indices[assay][bin_]])
        
        
#import predictions
df_dnn_single = pd.read_csv('mowal/Imputation_Paper/Results/predictions_dnn_assay_based_Ames.csv')
df_dnn_fn = pd.read_csv('mowal/Imputation_Paper/Results/predictions_dnn_fn_assay_based_Ames.csv')
df_xgb_single = pd.read_csv('mowal/Imputation_Paper/Results/predictions_xgb_assay_based_Ames.csv')
df_xgb_fn = pd.read_csv('mowal/Imputation_Paper/Results/predictions_xgb_fn_assay_based_Ames.csv')
df_rf_single = pd.read_csv('mowal/Imputation_Paper/Results/predictions_rf_assay_based_Ames.csv')
df_rf_fn = pd.read_csv('mowal/Imputation_Paper/Results/predictions_rf_fn_assay_based_Ames.csv')
df_dnn_mt = pd.read_csv('mowal/Imputation_Paper/Results/predictions_dnn_mt_assay_based_Ames.csv')
df_macau = pd.read_csv('mowal/Imputation_Paper/Results/predictions_macau_assay_based_Ames.csv')

#method -> df
df_dict = {'dnn':df_dnn_single,'dnn_fn':df_dnn_fn,'xgb':df_xgb_single, 'xgb_fn':df_xgb_fn, 'rf':df_rf_single, 'rf_fn':df_rf_fn, 'dnn_mt': df_dnn_mt,
           'macau':df_macau}

#get dict with predictions
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
            
            for bin_ in ['0-1','2-3','>3']:
                assay_bin_indices = dict_numbers_assay_bin_indices[assay][bin_]
                dict_predictions[method][assay][round_][bin_] = list(df_method_assay_round[df_method_assay_round['test_index'].isin(assay_bin_indices)]['prediction'])
                
                
                
#store results (mcc scores) in df with cols: method, assay, round, bin 
method_list = []
assay_list = []
round_list = []
bin_list = []
mcc_list = []

#compute MCC for each method - round - bin combination
for method in ['dnn','dnn_fn','xgb','xgb_fn','rf','rf_fn','dnn_mt','macau']:
    for assay in assays:
        for round_ in range(20):
            for bin_ in ['0-1','2-3','>3']:
                method_list.append(method)
                assay_list.append(assay)
                round_list.append(round_)
                bin_list.append(bin_)
                mcc_list.append(matthews_corrcoef(dict_numbers_assay_bin_true[assay][bin_],
                                                  dict_predictions[method][assay][round_][bin_]))

df_scores = pd.DataFrame(data={'method':method_list,'assay':assay_list,'round':round_list,'bin':bin_list,'MCC':mcc_list})
df_scores.to_csv('mowal/Imputation_Paper/Results/mcc_data_availability_bins_Ames.csv',index=False)                