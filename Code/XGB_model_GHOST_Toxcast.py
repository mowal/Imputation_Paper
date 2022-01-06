# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:11:14 2021

@author: Moritz
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import xgboost as xgb
from GHOST import ghostml
import math

#function to get binary predictions from model outputs according to threshold selected with GHOST
def probs_to_binary_with_threshold(probs,threshold):
    y_pred = np.array([1 if i >threshold else 0 for i in probs])
    return(y_pred)



#train and test set for assay-based splits imported
df_train = pd.read_csv('mowal/Imputation_Paper/Data/Train_Test_Splits/train_set_assay_based_Toxcast.csv')
df_test = pd.read_csv('mowal/Imputation_Paper/Data/Train_Test_Splits/test_set_assay_based_Toxcast.csv')


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
prediction_ghost_col = []
threshold_col = []


#import hyperparameters from .csv file
df_hyperparameters = pd.read_csv('mowal/Imputation_Paper/Data/Hyperparameters/XGB_huperparameetrs.csv')
df_hyperparameters_row = df_hyperparameters[df_hyperparameters['dataset']=='Toxcast']

n_rounds = df_hyperparameters_row['n_rounds'].iloc[0]
eta = df_hyperparameters_row['eta'].iloc[0]
colsample_bytree = df_hyperparameters_row['colsample_bytree'].iloc[0]
lam = df_hyperparameters_row['lambda'].iloc[0]
alpha = df_hyperparameters_row['alpha'].iloc[0]
scale_pos_weight = df_hyperparameters_row['scale_pos_weight'].iloc[0]


#iterate through assays
for i,assay in enumerate(assays):
    print(i,assay)
    
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
            
    
    
        
           
    #only 1 random seed
    for seed_round in range(1):
       
        
        
        param_dict = {'eta':eta,'colsample_bytree':colsample_bytree,'lambda':lam,'alpha':alpha,'scale_pos_weight':scale_pos_weight,'tree_method':'gpu_hist','objective':'binary:logistic',
                      'seed':seed_round}
        #train model
        dtrain = xgb.DMatrix(X_train,label=y_train)
        dtest = xgb.DMatrix(X_test)
                
        clf = xgb.train(param_dict,dtrain,n_rounds)
        
        thresholds = np.round(np.arange(0.05,0.55,0.05),2)
        y_train_probs = clf.predict(dtrain)
        #get optimised thresholds using GHOST
        threshold_opt = ghostml.optimize_threshold_from_predictions(y_train, y_train_probs, thresholds, ThOpt_metrics = 'ROC')
        
        
        #get predictions for df_predictions file
        assay_col_round = [assay for i in range(len(dict_indices_test[assay]))]
        assay_col+=assay_col_round
        
        round_col_round = [seed_round for i in range(len(dict_indices_test[assay]))]
        round_col+=round_col_round
        
        test_index_col+= dict_indices_test[assay]
        
        predictions_proba = list(clf.predict(dtest))
        prediction_proba_col+= predictions_proba
        
        predictions = [1 if i>0.5 else 0 for i in predictions_proba]
        prediction_col+= predictions
        
        predictions_ghost = list(probs_to_binary_with_threshold(predictions_proba,threshold_opt))
        prediction_ghost_col+= predictions_ghost
        
        thresholds = [threshold_opt for i in range(len(dict_indices_test[assay]))]
        threshold_col+=thresholds

#create and export predictions for single compounds
df_predictions = pd.DataFrame(data={'assay':assay_col,'round':round_col,'test_index':test_index_col,'prediction':prediction_col,
                                    'prediction_proba':prediction_proba_col,'prediction_ghost':prediction_ghost_col,'threshold':threshold_col})
df_predictions.to_csv('mowal/Imputation_Paper/Results/predictions_xgb_toxcast.csv',index=False)
