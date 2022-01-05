# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 11:45:07 2022

@author: Moritz
"""

import pandas as pd


df_toxcast = pd.read_csv('mowal/Imputation_Paper/Data/Datasets/Toxcast_aggregated.csv')

#keep all assays with at least 50 actives and 50 inactives
assays_to_keep = []

for col in df_toxcast.columns[:-1]:
    counts = df_toxcast[col].value_counts()
    if counts[0] >=50 and counts[1] >=50:
        assays_to_keep.append(col)
        
assays_to_keep.append('standardised_smiles')

df_toxcast_filtered = df_toxcast.loc[:,assays_to_keep].copy()

df_toxcast_filtered.to_csv('mowal/Imputation_Paper/Data/Datasets/Toxcast_filtered.csv',index=False)