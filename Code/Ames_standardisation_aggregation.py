# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:03:42 2021

@author: Moritz
"""
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

from rdkit import Chem

from molvs.metal import MetalDisconnector
from molvs.fragment import FragmentRemover
from molvs.normalize import Normalizer
from molvs.standardize import canonicalize_tautomer_smiles
from molvs.charge import Uncharger

import pandas as pd
import numpy as np
import math

#functions
def valid_smiles(smiles):
    #checks if valid rdkit mol can be obtained; rdkit constructs a valid mol from an empty string, mark them as invalid
    #return True or False
    m = Chem.MolFromSmiles(smiles)
    if smiles == '':
        return(False)
    elif m == None:
        return(False)
    elif m!= None:
        return(True)
        
def disconnect_metal(smiles):
    #return smiles with disconnected metaks
    m = Chem.MolFromSmiles(smiles)
    md = MetalDisconnector()
    m = md.disconnect(m)
    return(Chem.MolToSmiles(m))

def is_organic(fragment):
    #check if mol is organic
    for a in fragment.GetAtoms():
        if a.GetAtomicNum() == 6:
            return True
    return False

def fragment_removal_smiles(smiles, leave_last=False):
    #Utility function that returns the result SMILES after FragmentRemover is applied to given a SMILES string
    #if only single organic fragment: don't remove anything
    #elif only one of the fragments organic: keep only organic
    #else: remove specified fragments
    mol = Chem.MolFromSmiles(smiles)
    fragments = Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=False)
    fragment_count = len(fragments)
    if fragment_count == 1 and is_organic(fragments[0]) == True:
        return(smiles)
    
    else:
        organic_frags = []
        for i,fragment in enumerate(fragments):
            if is_organic(fragment)==True:
                organic_frags.append(i)

        organic_count = len(organic_frags)

        if organic_count == 1:
            return(Chem.MolToSmiles(fragments[organic_frags[0]]))

        else:
            mol = FragmentRemover(leave_last=leave_last).remove(mol)
            return(Chem.MolToSmiles(mol, isomericSmiles=True))

def normalize_smiles(smiles):
    """Utility function that runs normalization rules on a given a SMILES string."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = Normalizer().normalize(mol)
    if mol:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    

def uncharge_smiles(smiles):
    """Utility function that returns the uncharged SMILES for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    u = Uncharger()
    mol = u.uncharge(mol)
    if mol:
        return(Chem.MolToSmiles(mol, isomericSmiles=True))

def neutralizeRadicals(smiles):
    #neutralise mols that are charged because being radicals
    mol = Chem.MolFromSmiles(smiles)
    for a in mol.GetAtoms():
        if a.GetNumRadicalElectrons()>=1 and a.GetFormalCharge()>=1:
            a.SetNumRadicalElectrons(0)         
            a.SetFormalCharge(0)
    return(Chem.MolToSmiles(mol))

def standardise(smiles):
    #remove Hs
    smiles = Chem.MolToSmiles(Chem.rdmolops.RemoveHs(Chem.MolFromSmiles(smiles)))
    #disconnect metals
    smiles = disconnect_metal(smiles)
    #remove fragments
    smiles = fragment_removal_smiles(smiles)
    #uncharge
    smiles = uncharge_smiles(smiles)
    #uncharge radicals
    smiles = neutralizeRadicals(smiles)
    #normalise chemotypes
    smiles = normalize_smiles(smiles)
    #canonicalise tautomers
    smiles = canonicalize_tautomer_smiles(smiles)
    #transform to rdkit mol and back to get rdkit canonical version
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles,sanitize=True),canonical=True)
    #transform to inchi key and back to canonicalise tautomers not considered by molvs
    final_smiles = Chem.MolToSmiles(Chem.MolFromInchi(Chem.MolToInchi(Chem.MolFromSmiles(smiles,sanitize=True))),canonical=True)
    #if final_smiles different than smiles return smiles as well, return as a list
    if final_smiles == smiles:
        return([final_smiles])
    else:
        return([final_smiles,smiles])

def smiles_to_frags(smiles):
    #convert smiles to list of frags
    frags = Chem.GetMolFrags(Chem.MolFromSmiles(smiles), asMols=True)
    fraglist = [Chem.MolToSmiles(frag) for frag in frags]
    return(fraglist)

def drop_fragments(smiles):
#handling of multiple fragments:
# remove inorganic fragments
#if identical/stereoisomers: keep only 1
#if different organic components: drop

#handling of single fragments
#if not organic: return ''
#else: return smiles

    if len(smiles_to_frags(smiles))>1:
        smileslist = smiles_to_frags(smiles)
        #remove all inorganic fragments, if list becomes empty: return ''
        organics = []
        for frag in smileslist:
            if is_organic(Chem.MolFromSmiles(frag)) == True:
                organics.append(frag)
        if len(organics) == 0:
            return('')
        elif len(organics) == 1:
            return(organics[0])
        else:
            smileslist = organics
            #check if all frags are identical, if all identical, return first
            identical = True
            for fra in smileslist[1:]:
                if fra != smileslist[0]:
                    identical=False
            if identical == True:
                return(smileslist[0])
            else:
                return('')
        
    
    else:
        if is_organic(Chem.MolFromSmiles(smiles)) == True:
            return(smiles)
        else:
            return('')


def majority_agg(df_grouped): #function that performs aggregation of several rows in the a df grouped by identical standardised SMILES
    #store aggregated rows in list
    df_records = []
    for smiles in df_grouped.groups.keys():
        df = df_grouped.get_group(smiles)
        if df.shape[0] == 1:
            df_records.append(df)
        else:
            d = {}
            for col in df.columns[:-1]:
                act=0
                inact=0
                for i in df[col]:
                    if i==1:
                        act+=1
                    elif i==0:
                        inact+=1
                
                if act > inact:
                    d[col] = 1
                elif act < inact:
                    d[col] = 0
                else:
                    d[col] = np.nan
            
            #df that contains agg values for single smile
            df_smi_agg = pd.DataFrame(d,index=[0])
            df_smi_agg['standardised_smiles'] = smiles
            #collect single-line dfs in list
            df_records.append(df_smi_agg)
    df_return = pd.concat(df_records)
    return(df_return)


df = pd.read_csv('mowal/Imputation_Paper/Data/Datasets/ISSSTY_v1a_7367_02May011.txt',delimiter='\t')

df.drop(labels=['Structure [idcode]','Unnamed: 72','SMILES'],axis=1,inplace=True)
df.rename(columns={'Smiles_new': 'Smiles'},inplace=True)

#drop rows for which no mol can be generated in rdkit
df = df.astype({'Smiles' : 'str'})
valid_smis = []

for smi in df['Smiles']:
    valid_smis.append(valid_smiles(smi))
    
df['valid_smiles'] = valid_smis
df_valid = df[df['valid_smiles']==True].copy()
df_valid.drop('valid_smiles',axis=1,inplace=True)

standardised_smiles = []
failed = []
inchi_tautomers = []
for i,smi in enumerate(df_valid['Smiles']):
    print(i)
    try:
        standardised = standardise(smi)
        standardised_smiles.append(standardised[0])
        if len(standardised) ==2:
            inchi_tautomers.append(standardised)
    except:
        standardised_smiles.append('')
        failed.append((i,smi))

filtered_smiles = []
for smi in standardised_smiles:
    filtered_smiles.append(drop_fragments(smi))

df_valid['standardised_smiles'] = filtered_smiles
df_filtered = df_valid[df_valid['standardised_smiles']!=''].copy()


#aggregate identical SMILES in df_filtered

#create dictionary standardised_smiles --> CAS to keep track from which mols the entries were aggregated
dict_smi_cas = {}
for i,row in df_filtered.iterrows():
    if row['standardised_smiles'] not in dict_smi_cas:
        dict_smi_cas[row['standardised_smiles']] = [row['CAS']]
    else:
        dict_smi_cas[row['standardised_smiles']].append(row['CAS'])



#columns to keep
col_keep = ['TA1535', 'TA1537', 'TA100', 'TA100_S9', 'TA98', 'TA98_S9', 'TA1535_S9',
                       'TA1537_S9', 'TA102', 'TA102_S9', 'TA97', 'TA97_S9', 'standardised_smiles']

col_indices = []
for i,col in enumerate(df_filtered.columns):
    if col in col_keep:
        col_indices.append(i)
        
#keep only interesting strains
df_inter = df_filtered.iloc[:,col_indices].copy()

#map 'ND' and 2 to math.nan, map 1 to 0, map 3 to 1

df_inter.replace('ND', math.nan, inplace=True)
df_inter.replace('2', math.nan, inplace=True)
df_inter.replace('1', 0, inplace=True)
df_inter.replace('3', 1, inplace=True)

df_grouped = df_inter.groupby('standardised_smiles')
df_agg = majority_agg(df_grouped=df_grouped)
df_agg.reset_index(inplace=True,drop=True)

#add column containing CAS of original entries that map to the aggregated data row
df_agg['CAS'] = [dict_smi_cas[smi] for smi in df_agg['standardised_smiles']]

df_agg.to_csv('mowal/Imputation_Paper/Data/Datasets/Ames_aggregated.csv',index=False)
