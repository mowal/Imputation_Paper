# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:29:03 2021

@author: Moritz
"""
import tensorflow as tf
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import math

#define functions to define, compile and train DNNs
def define_dnn(hidden_l,neurons_per_l,dropout,l2_ker,in_dim=2048): #for Step 2 of FN models: in_dim needs to be changed to 2059 (chemical descriptors and 11 remaining assays)
    model = Sequential()
    model.add(Dense(units=neurons_per_l, activation='relu',input_dim=in_dim,kernel_regularizer=l2(l2_ker)))
    if dropout > 0:
        model.add(Dropout(dropout))
    for i in range(hidden_l-1):
        model.add(Dense(units=neurons_per_l, activation='relu',kernel_regularizer=l2(l2_ker)))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(units=1, activation='sigmoid'))
    return(model)
    
def compile_dnn(model,lr):
    model.compile(Adam(lr=lr),loss='binary_crossentropy',metrics=['accuracy'])
    return(model)
    

    
def train_dnn(model,X_train,y_train,class_weight,batch,epochs):
    
    model.fit(X_train,y_train, validation_split=0.0,batch_size=batch,class_weight=class_weight, epochs=epochs, shuffle=True,verbose=0)
    return(model)
    
#train and test set for assay-based splits imported
df_train = pd.read_csv('mowal/Imputation_Paper/Data/train_set_assay_based_Ames.csv')
df_test = pd.read_csv('mowal/Imputation_Paper/Data//test_set_assay_based_Ames.csv')
df_params = pd.read_csv('mowal/Imputation_Paper/Data/DNN_single_hyperparameters.csv')

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
    hil = int(df_params_slice.loc[assay,'hidden_layers'])
    npl = int(df_params_slice.loc[assay,'nodes_per_layer'])
    ler = float(df_params_slice.loc[assay,'learning_rate'])
    dro = float(df_params_slice.loc[assay,'dropout'])
    l2r = float(df_params_slice.loc[assay,'L2_regulation'])
    bat = int(df_params_slice.loc[assay,'batch_size'])
    clw = float(df_params_slice.loc[assay,'class_weight'])
    epo = float(df_params_slice.loc[assay,'epochs'])
    
        
    #iterate through 20 rounds with different random seeds
    for seed_round in range(20):
        print(assay,seed_round)
        
        tf.random.set_seed(seed_round)
        
        model = define_dnn(hidden_l=hil,neurons_per_l=npl,dropout=dro,l2_ker=l2r)
        model = compile_dnn(model=model,lr=ler)
        model = train_dnn(model=model,X_train=X_train,y_train=y_train,
                                      class_weight=clw,batch=bat,epochs=epo)
        
        #get predictions for df_predictions file (all rounds)
        assay_col_round = [assay for i in range(len(dict_indices_test[assay]))]
        assay_col+=assay_col_round
        
        round_col_round = [seed_round for i in range(len(dict_indices_test[assay]))]
        round_col+=round_col_round
        
        test_index_col+= dict_indices_test[assay]
        
        predictions_proba = list(model.predict(X_test))
        prediction_proba_col+= predictions_proba
        
        predictions = [1 if i>0.5 else 0 for i in predictions_proba]
        prediction_col+= predictions
        
        tf.keras.backend.clear_session()
        



#create and export predictions for single compounds
df_predictions = pd.DataFrame(data={'assay':assay_col,'round':round_col,'test_index':test_index_col,'prediction':prediction_col,
                                    'prediction_proba':prediction_proba_col})
df_predictions.to_csv('mowal/Imputation_Paper/Data/predictions_dnn_assay_based_Ames.csv',index=False)