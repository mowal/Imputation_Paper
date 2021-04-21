# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 08:52:29 2021

@author: Moritz
"""

import pandas as pd
import numpy as np
import math

def cont_table_assays(assay1,assay2):
    #get contigengency table that decribes actives in assay1 and assay2
    #considers only substances measured in both assays, return number of overlapping compounds
    #return 2x2 contingengy table: (0,0)/a: A+B+, (0,1)/b: A+B-, (1,0)/c: A-B+, (1,1)/d: A-B-
    
    a = 0
    b = 0
    c = 0
    d = 0
    
    for ass1,ass2 in zip(assay1,assay2):
        
        if math.isnan(ass1) or math.isnan(ass2):
            continue
        
        if ass1 == 1 and ass2 == 1:
            a+=1
        
        elif ass1 == 1 and ass2 == 0:
            b+=1
        
        elif ass1 == 0 and ass2 == 1:
            c+=1
        
        elif ass1 == 0 and ass2 == 0:
            d+=1
            
    return(np.array([[a,b],[c,d]]))    
    
def entrop1(array):
    a = array[0,0]
    b = array[0,1]
    c = array[1,0]
    d = array[1,1]
    
    #prob active in assay1
    p1 = (a+b)/(a+b+c+d)
    #entropy assay1
    h1 = -p1*np.log2(p1) - (1-p1)*np.log2(1-p1)
    return(h1)
    
def entrop2(array):
    a = array[0,0]
    b = array[0,1]
    c = array[1,0]
    d = array[1,1]
    
     #prob active in assay2
    p2 = (a+c)/(a+b+c+d)
    #entropy assay2
    h2 = -p2*np.log2(p2) - (1-p2)*np.log2(1-p2)
    return(h2)    

def joint_entrop(array):
    a = array[0,0]
    b = array[0,1]
    c = array[1,0]
    d = array[1,1]

    #prob each table field
    p3 = a/(a+b+c+d)
    p4 = b/(a+b+c+d)
    p5 = c/(a+b+c+d)
    p6 = d/(a+b+c+d)
    
    #joint entropy
    h12 = -p3*np.log2(p3) - p4*np.log2(p4) - p5*np.log2(p5) - p6*np.log2(p6)
    return(h12)

def mutual_info(array):
    a = array[0,0]
    b = array[0,1]
    c = array[1,0]
    d = array[1,1]
    
    #prob active in assay1
    p1 = (a+b)/(a+b+c+d)
    #entropy assay1
    h1 = -p1*np.log2(p1) - (1-p1)*np.log2(1-p1)
    
    #prob active in assay2
    p2 = (a+c)/(a+b+c+d)
    #entropy assay2
    h2 = -p2*np.log2(p2) - (1-p2)*np.log2(1-p2)
    
    #prob each table field
    p3 = a/(a+b+c+d)
    p4 = b/(a+b+c+d)
    p5 = c/(a+b+c+d)
    p6 = d/(a+b+c+d)
    
    #joint entropy
    h12 = -p3*np.log2(p3) - p4*np.log2(p4) - p5*np.log2(p5) - p6*np.log2(p6)
    
    #mutual information
    mi = h1 + h2 - h12
    return(mi)
    
    
df = pd.read_csv('mowal/Imputation_Paper/Data/Datasets/Ames_aggregated.csv')
df.head()

df.drop('standardised_smiles',axis=1,inplace=True)

#get all assay combinations
assay_combs = []
for i in df.columns:
    for j in df.columns:
        #only different assays
        if i==j:
            continue
        #each combination only once
        if '{}|{}'.format(j,i) in assay_combs:
            continue
        
        assay_combs.append('{}|{}'.format(i,j))
        
#list metrics for all combinations all metrics

minfo = []
ent1 = []
ent2 = []
jent = []
mi_joint = []
mi_single1 = []
mi_single2 = []

for comb in assay_combs:
    assay1 = df[comb.split('|')[0]]
    assay2 = df[comb.split('|')[1]]
    cont_table = cont_table_assays(assay1,assay2)
    ent1.append(entrop1(cont_table))
    ent2.append(entrop2(cont_table))
    jent.append(joint_entrop(cont_table))
    minfo.append(round(mutual_info(cont_table),5))
    mi_joint.append(minfo[-1]/jent[-1])
    mi_single1.append(minfo[-1]/ent1[-1])
    mi_single2.append(minfo[-1]/ent2[-1])
    
    
    
  
#entropy1/2: entropy of first/second assay, mi/entropy1: ratio MI and entropy of 1st assay, mi/entropy2: ratio MI and entropy of 2nd assay, mutual_info: MI 1st abd 2bd assay
#joint_entropy: joint entropy 1st and second, mi/joint: ratio MI and joint entropy
df_results = pd.DataFrame(data={'assay_combination':assay_combs, 'entropy1':ent1, 'mi/entropy1': mi_single1, 'entropy2':ent2, 'mi/entropy2':mi_single2,
                                'mutual_info':minfo, 'joint_entropy':jent,'mi/joint':mi_joint})


df_results.to_csv('mowal/Imputation_Paper/Results/info_metrics_Ames.csv',index=False)
