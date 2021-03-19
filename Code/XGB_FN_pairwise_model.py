# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:02:01 2021

@author: Moritz
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb
import math

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

#train and test set for assay-based splits imported
df_train = pd.read_csv('mowal/Imputation_Paper/Data/train_set_assay_based_Ames.csv')
df_test = pd.read_csv('mowal/Imputation_Paper/Data/test_set_assay_based_Ames.csv')
df_filled = df_train.iloc[:,:-1].copy()
df_params = pd.read_csv('mowal/Imputation_Paper/Data/XGB_hyperparameters.csv')

#get correct slice of df_params (Ames, assay-based splits)
df_params_slice = df_params[(df_params['dataset']=='Ames')&(df_params['split']=='assay')].copy()
df_params_slice.set_index('assay',inplace=True)

#get indices of train and test compounds for each assay
assays = df_train.columns[:-1]
dict_indices_train = {}
for assay in assays:
    dict_indices_train[assay] = []
    for i,j in enumerate(df_train[assay]):
        if not math.isnan(j):
            dict_indices_train[assay].append(i)

dict_indices_test = {}
for assay in assays:
    dict_indices_test[assay] = []
    for i,j in enumerate(df_test[assay]):
        if not math.isnan(j):
            dict_indices_test[assay].append(i)
            

#create X matrix for all compounds to fill gaps
X_all = np.empty([df_train.shape[0],2048])
smis = df_train['standardised_smiles'].tolist()
mols = [Chem.MolFromSmiles(smile) for smile in smis]
fps_bit = [AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=2048) for mol in mols]
    
for i,j in enumerate(fps_bit):
    for k,l in enumerate(list(j)):
        X_all[i,k] = l

#STEP1 fill gaps with single task models
for assay in assays:
    
    #get X_train and y_train matrix for assay
    df_assay_train = pd.concat([df_train[assay],df_train['standardised_smiles']],axis=1)
    df_assay_train.dropna(inplace=True)
    
    y_train = df_assay_train[assay]
    
    X_train = np.empty([df_assay_train.shape[0],2048])
    
    smis = df_assay_train['standardised_smiles'].tolist()
    mols = [Chem.MolFromSmiles(smile) for smile in smis]
    fps_bit = [AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=2048) for mol in mols]
    
    for i,j in enumerate(fps_bit):
        for k,l in enumerate(list(j)):
            X_train[i,k] = l
                      
    
    #get hyperparameters for assay
    
    nr = int(df_params_slice.loc[assay,'n_rounds'])
    et = float(df_params_slice.loc[assay,'eta'])
    co = float(df_params_slice.loc[assay,'colsample_bytree'])
    la = int(df_params_slice.loc[assay,'lambda'])
    al = int(df_params_slice.loc[assay,'alpha'])
    if df_params_slice.loc[assay,'scale_pos_weight'] == 'weight':
        sc = sc= (y_train.shape[0]-sum(y_train))/sum(y_train)
    else:
        sc = 1
        
    param_dict = {'eta':et,'colsample_bytree':co,'lambda':la,'alpha':al,'scale_pos_weight':sc,'tree_method':'gpu_hist','objective':'binary:logistic',
                      'seed':0}
        
    #train model
    dtrain = xgb.DMatrix(X_train,label=y_train)
    dfill = xgb.DMatrix(X_all)
        
    clf = xgb.train(param_dict,dtrain,nr)
        
        
    #create predictions for data filling
    predictions_filling_proba = list(clf.predict(dfill))
    predictions_filling = [1 if i>0.5 else 0 for i in predictions_filling_proba]
        
    #fill gaps in df_filled with predictions
    for i,row in df_filled.iterrows():
        if math.isnan(row[assay]):
            df_filled.loc[i,assay] = predictions_filling[i]


#STEP2: models with related assays as additional features 

#create lists for final df_scores
assay1_col = []
assay2_col = []
round_col = []
acc = []
bal_acc = []
prec = []
rec = []
auc = []
f1 = []
mcc = []

#iterate through target assays
for assay in assays:
    
    #get X_train and y_train matrix for assay
    df_assay_train = pd.concat([df_train[assay],df_train['standardised_smiles']],axis=1)
    df_assay_train.dropna(inplace=True)
    
    y_train = df_assay_train[assay]
    
    X_train_fp = np.empty([df_assay_train.shape[0],2048])
    
    smis = df_assay_train['standardised_smiles'].tolist()
    mols = [Chem.MolFromSmiles(smile) for smile in smis]
    fps_bit = [AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=2048) for mol in mols]
    
    for i,j in enumerate(fps_bit):
        for k,l in enumerate(list(j)):
            X_train_fp[i,k] = l
    
    
    
    #get X_test and y_test matrix for assay
    df_assay_test = pd.concat([df_test[assay],df_test['standardised_smiles']],axis=1)
    df_assay_test.dropna(inplace=True)
    
    y_test = df_assay_test[assay]
    
    X_test_fp = np.empty([df_assay_test.shape[0],2048])
    
    smis = df_assay_test['standardised_smiles'].tolist()
    mols = [Chem.MolFromSmiles(smile) for smile in smis]
    fps_bit = [AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=2048) for mol in mols]
    
    for i,j in enumerate(fps_bit):
        for k,l in enumerate(list(j)):
            X_test_fp[i,k] = l
            
        
    #get hyperparameters for target assay
    
    nr = int(df_params_slice.loc[assay,'n_rounds'])
    et = float(df_params_slice.loc[assay,'eta'])
    co = float(df_params_slice.loc[assay,'colsample_bytree'])
    la = int(df_params_slice.loc[assay,'lambda'])
    al = int(df_params_slice.loc[assay,'alpha'])
    if df_params_slice.loc[assay,'scale_pos_weight'] == 'weight':
        sc = sc= (y_train.shape[0]-sum(y_train))/sum(y_train)
    else:
        sc = 1
    
    #iterate through auxiliary assays     
    for assay2 in assays:
        
        if assay2 == assay:
            continue
        
        #get X_train and X_test by concatenating chemical feature matrices with labels of assay2
        feature_vector_train = np.array(df_filled.loc[dict_indices_train[assay],assay2]).reshape(len(dict_indices_train[assay]),1)
        X_train = np.concatenate((X_train_fp,feature_vector_train),axis=1)
        
        #get X_test
        feature_vector_test = np.array(df_filled.loc[dict_indices_test[assay],assay2]).reshape(len(dict_indices_test[assay]),1)
        X_test = np.concatenate((X_test_fp,feature_vector_test),axis=1)
        
        dtrain = xgb.DMatrix(X_train,label=y_train)
        dtest = xgb.DMatrix(X_test)
    
    
    
        #iterate through 20 rounds with different random seeds
        for seed_round in range(20):
            print(assay,seed_round)
            
            param_dict = {'eta':et,'colsample_bytree':co,'lambda':la,'alpha':al,'scale_pos_weight':sc,'tree_method':'gpu_hist','objective':'binary:logistic',
                          'seed':seed_round}
            
            
            #train model
            clf = xgb.train(param_dict,dtrain,nr)
            
            y_pred_proba = clf.predict(dtest)
            y_pred = np.array([1 if i>0.5 else 0 for i in y_pred_proba])
            
    
            #get predictions for df_predictions file (all rounds)
            assay1_col_round = [assay for i in range(len(dict_indices_test[assay]))]
            assay1_col+=assay1_col_round
            
            assay2_col_round = [assay for i in range(len(dict_indices_test[assay]))]
            assay2_col+=assay2_col_round
            
            round_col_round = [seed_round for i in range(len(dict_indices_test[assay]))]
            round_col+=round_col_round
            
            acc.append(accuracy_score(y_test,y_pred))
            bal_acc.append(balanced_accuracy_score(y_test,y_pred))
            prec.append(precision_score(y_test,y_pred))
            rec.append(recall_score(y_test,y_pred))
            auc.append(roc_auc_score(y_test,y_pred_proba))
            f1.append(f1_score(y_test,y_pred))
            mcc.append(matthews_corrcoef(y_test,y_pred))
            
               

#create and export scores
df_scores = pd.DataFrame(data={'target_assay':assay1_col,'auxiliary_assay':assay2_col,'round':round_col,'accuracy':acc,'balanced_accuracy':bal_acc,'precision':prec,
                               'recall':rec,'AUC':auc,'F1':f1,'MCC':mcc})

df_scores.to_csv('mowal/Imputation_Paper/Results/scores_xgb_fn_pairwise_Ames.csv',index=False)