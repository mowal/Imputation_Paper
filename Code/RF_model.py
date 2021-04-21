# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:11:14 2021

@author: Moritz
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
import math

#train and test set for assay-based splits imported
df_train = pd.read_csv('mowal/Imputation_Paper/Data/Train_Test_Splits/train_set_assay_based_Ames.csv')
df_test = pd.read_csv('mowal/Imputation_Paper/Data/Train_Test_Splits/test_set_assay_based_Ames.csv')
df_params = pd.read_csv('mowal/Imputation_Paper/Data/Hyperparameters/RF_hyperparameters.csv')

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
    if df_params_slice.loc[assay,'weight'] == 'None':
        cw = None
    else:
        cw = df_params_slice.loc[assay,'weight']
    trees = int(df_params_slice.loc[assay,'trees'])
    if df_params_slice.loc[assay,'max_features'] == '0.25':
        mf = 0.25
    else:
        mf = df_params_slice.loc[assay,'max_features']
    
           
    #iterate through 20 rounds with different random seeds
    for seed_round in range(20):
        print(assay,seed_round)
        
        
        clf = RandomForestClassifier(class_weight=cw, n_estimators=trees, max_features=mf,random_state=seed_round)
        
        #fit model
        clf.fit(X_train, y_train)
        
        
        #get predictions for df_predictions file (all rounds)
        assay_col_round = [assay for i in range(len(dict_indices_test[assay]))]
        assay_col+=assay_col_round
        
        round_col_round = [seed_round for i in range(len(dict_indices_test[assay]))]
        round_col+=round_col_round
        
        test_index_col+= dict_indices_test[assay]
        
        predictions_proba = list(clf.predict_proba(X_test)[:,1])
        prediction_proba_col+= predictions_proba
        
        predictions = list(clf.predict(X_test))
        prediction_col+= predictions
        
        

#create and export predictions for single compounds
df_predictions = pd.DataFrame(data={'assay':assay_col,'round':round_col,'test_index':test_index_col,'prediction':prediction_col,
                                    'prediction_proba':prediction_proba_col})
df_predictions.to_csv('mowal/Imputation_Paper/Results/Predictions/predictions_rf_assay_based_Ames.csv',index=False)
