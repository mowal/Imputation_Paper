# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:55:41 2021

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
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.backend import cast
from tensorflow.keras.backend import not_equal
from tensorflow.keras.backend import floatx
import math

#define functions to define, compile and train DNNs
def define_dnn_mt(hidden_l,neurons_per_l,dropout,l2_ker,in_dim=2048):
    model = Sequential()
    model.add(Dense(units=neurons_per_l, activation='relu',input_dim=in_dim,kernel_regularizer=l2(l2_ker)))
    if dropout > 0:
        model.add(Dropout(dropout))
    for i in range(hidden_l-1):
        model.add(Dense(units=neurons_per_l, activation='relu',kernel_regularizer=l2(l2_ker)))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(units=12, activation='sigmoid'))
    return(model)

def masked_loss_function(y_true, y_pred):
    mask = cast(not_equal(y_true, -1), floatx())
    return binary_crossentropy(y_true * mask, y_pred * mask)

def compile_dnn_mt(model,lr):
    model.compile(Adam(lr=lr),loss=masked_loss_function,metrics=['accuracy'])
    return(model)
    

    
def train_dnn(model,X_train,y_train,class_weight,batch,epochs):
    
    model.fit(X_train,y_train, validation_split=0.0,batch_size=batch,class_weight=class_weight, epochs=epochs, shuffle=True,verbose=0)
    return(model)
    

    
#train and test set for assay-based splits imported
df_train = pd.read_csv('mowal/Imputation_Paper/Data/train_set_assay_based_Ames.csv')
df_test = pd.read_csv('mowal/Imputation_Paper/Data/test_set_assay_based_Ames.csv')
df_params = pd.read_csv('mowal/Imputation_Paper/Data/DNN_multi_hyperparameters.csv')

#get correct slice of df_params (Ames, assay-based splits)
df_params_slice = df_params[(df_params['dataset']=='Ames')&(df_params['split']=='assay')].copy()


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

#fill in -1 as dummy for datagaps
df_train = df_train.fillna(-1)
df_test = df_test.fillna(-1)

#generate X and y_train, y_test, note: X is identical for train and test
#generate ECFP and store in X
smis = df_train['standardised_smiles'].tolist()
mols = [Chem.MolFromSmiles(smile) for smile in smis]
fps_bit = [AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=2048) for mol in mols]
    
X = np.empty([df_train.shape[0],2048])
for i,j in enumerate(fps_bit):
    for k,l in enumerate(list(j)):
        X[i,k] = l

y_train = np.array(df_train.iloc[:,:-1])
y_test = np.array(df_test.iloc[:,:-1])

#get hyperparameters
hil = int(df_params_slice['hidden_layers'][0])
npl = int(df_params_slice['nodes_per_layer'][0])
ler = float(df_params_slice['learning_rate'][0])
dro = float(df_params_slice['dropout'][0])
l2r = float(df_params_slice['L2_regulation'][0])
bat = int(df_params_slice['batch_size'][0])
clw = float(df_params_slice['class_weight'][0])
epo = float(df_params_slice['epochs'][0])


#create lists for final df_predictions
assay_col = []
round_col = []
test_index_col = []
prediction_col = []
prediction_proba_col = []

#iterate through 20 rounds with different random seeds
for seed_round in range(20):
    print(seed_round)
    tf.random.set_seed(seed_round)


    #train model
    model = define_dnn_mt(hidden_l=hil,neurons_per_l=npl,dropout=dro,l2_ker=l2r)
    model = compile_dnn_mt(model=model,lr=ler)
    model = train_dnn(model=model,X_train=X,y_train=y_train,
                                      class_weight=clw,batch=bat,epochs=epo)
                
    y_pred_proba = model.predict(X)
                
    #convert y_pred_proba to y_pred by rounding
    y_pred = np.round(y_pred_proba,0)
                
    #store predictions            
    for i,assay in enumerate(df_test.columns[:-1]):
        test_index_col+= dict_indices_test[assay]
        assay_col+= [assay for i in range(len(dict_indices_test[assay]))]
        round_col+= [seed_round for i in range(len(dict_indices_test[assay]))]
        prediction_col+= list(y_pred[dict_indices_test[assay],i])
        prediction_proba_col+= list(y_pred_proba[dict_indices_test[assay],i])

    tf.keras.backend.clear_session() 
    
#create and export
df_predictions = pd.DataFrame(data={'assay':assay_col,'round':round_col,'test_index':test_index_col,'prediction':prediction_col,
                                    'prediction_proba':prediction_proba_col})
df_predictions.to_csv('../results/dnn_mt/predictions_dnn_mt_assay_based_Ames.csv',index=False)
