# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:43:34 2021

@author: Moritz
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb
import math

#train and test set for assay-based splits imported
df_train = pd.read_csv('mowal/Imputation_Paper/Data/train_set_assay_based_Ames.csv')
df_test = pd.read_csv('mowal/Imputation_Paper/Data/test_set_assay_based_Ames.csv')
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



#create lists for final df_predictions
assay_col = []
round_col = []
test_index_col = []
prediction_col = []
prediction_proba_col = []


#iterate through assays
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
            
    #get X_test and y_test matrix for assay
    df_assay_test = pd.concat([df_test[assay],df_test['standardised_smiles']],axis=1)
    df_assay_test.dropna(inplace=True)
    
    y_test = df_assay_test[assay]
    
    X_test = np.empty([df_assay_test.shape[0],2048])
    
    smis = df_assay_test['standardised_smiles'].tolist()
    mols = [Chem.MolFromSmiles(smile) for smile in smis]
    fps_bit = [AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=2048) for mol in mols]
    
    for i,j in enumerate(fps_bit):
        for k,l in enumerate(list(j)):
            X_test[i,k] = l
            
    
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
        
    #iterate through 20 rounds with different random seeds
    for seed_round in range(20):
        print(assay,seed_round)
        
        
        param_dict = {'eta':et,'colsample_bytree':co,'lambda':la,'alpha':al,'scale_pos_weight':sc,'tree_method':'gpu_hist','objective':'binary:logistic',
                      'seed':seed_round}
        #train model
        dtrain = xgb.DMatrix(X_train,label=y_train)
        dtest = xgb.DMatrix(X_test)
                
        clf = xgb.train(param_dict,dtrain,nr)
        
        
               
        
        #get predictions for df_predictions file (all rounds)
        assay_col_round = [assay for i in range(len(dict_indices_test[assay]))]
        assay_col+=assay_col_round
        
        round_col_round = [seed_round for i in range(len(dict_indices_test[assay]))]
        round_col+=round_col_round
        
        test_index_col+= dict_indices_test[assay]
        
        predictions_proba = list(clf.predict(dtest))
        prediction_proba_col+= predictions_proba
        
        predictions = [1 if i>0.5 else 0 for i in predictions_proba]
        prediction_col+= predictions
        
        



#create and export predictions for single compounds
df_predictions = pd.DataFrame(data={'assay':assay_col,'round':round_col,'test_index':test_index_col,'prediction':prediction_col,
                                    'prediction_proba':prediction_proba_col})
df_predictions.to_csv('mowal/Imputation_Paper/Results/predictions_xgb_assay_based_Ames.csv',index=False)