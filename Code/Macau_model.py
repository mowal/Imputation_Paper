# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:13:56 2021

@author: Moritz
"""

import pandas as pd
import numpy as np
import scipy.sparse
import macau
from rdkit import Chem
from rdkit.Chem import AllChem


#train and test set for assay-based splits imported
df_train = pd.read_csv('mowal/Imputation_Paper/Data/Train_Test_Splits/train_set_assay_based_Ames.csv')
Y_np_train = df_train.iloc[:,:-1].values

#replace 0 by 0.001 to enable conversion to sparse matrix as required for the Macau model 
Y_np_train[Y_np_train < 0.5] = 0.001

Y_train = scipy.sparse.csr_matrix(Y_np_train)

smis = df_train['standardised_smiles'].tolist()
mols = [Chem.MolFromSmiles(smile) for smile in smis]
fps_bit = [AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=2048) for mol in mols]
    
X = np.empty([df_train.shape[0],2048])
for i,j in enumerate(fps_bit):
    for k,l in enumerate(list(j)):
        X[i,k] = l
        
ecfp = scipy.sparse.csr_matrix(X)

df_test = pd.read_csv('mowal/Imputation_Paper/Data/Train_Test_Splits/test_set_assay_based_Ames.csv')
Y_np_test = df_test.iloc[:,:-1].values
Y_np_test[Y_np_test < 0.5] = 0.001

Y_test = scipy.sparse.csr_matrix(Y_np_test)


#get hyperparameters
df_params = pd.read_csv('mowal/Imputation_Paper/Data/Hyperparameters/Macau_hyperparameters.csv')
df_params_slice = df_params[(df_params['dataset']=='Ames')&(df_params['split']=='assay')].copy()

nul = int(df_params_slice['num_latent'][0])
bur = int(df_params_slice['burnin'][0])
nsa = int(df_params_slice['nsamples'][0])


resultlist = []
#iterate thorough 20 different random seeds
for seed in range(20):

    #train model
    result = macau.macau(Y=Y_train,
                         Ytest=Y_test,
                         side=[ecfp,None],
                         num_latent=nul,
                         precision='probit',
                         univariate=False,
                         burnin=bur,
                         nsamples=nsa)
    
    result_df = result.prediction
    result_df['round'] = [seed for i in range(result_df.shape[0])]
    resultlist.append(result_df)

all_results = pd.concat(resultlist)
all_results.to_csv('mowal/Imputation_Paper/Results/Predictions/predictions_Macau_assay_based_Ames.csv')
