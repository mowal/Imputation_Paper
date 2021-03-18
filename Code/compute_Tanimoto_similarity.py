# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:48:41 2021

@author: Moritz
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import pandas as pd

#import file
df = pd.read_csv('mowal/Imputation_Paper/Data/Ames_aggregated.csv')


#generate list of fps from smiles
smiles = df['standardised_smiles'].tolist()
mols = [Chem.MolFromSmiles(smile) for smile in smiles]
fps_bit = [AllChem.GetMorganFingerprintAsBitVect(mol,2, nBits=2048) for mol in mols]

#write similarities to plain text file, note the .txt is due to its size not stored in the repository
with open('tanimoto_similarities_Ames.txt','w') as file:
    for count1,fp in enumerate(fps_bit):
        count2 = 0
        for count2,fp2 in enumerate(fps_bit):
           
            if count2 > count1:
                line = str(count1) + ',' + str(count2) + ',' + str(round(DataStructs.FingerprintSimilarity(fp, fp2),3)) + '\n'
                file.write(line)