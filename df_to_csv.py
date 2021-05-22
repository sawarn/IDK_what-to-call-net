#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 00:43:11 2020

@author: shivam
"""
import os
import pandas as pd
import numpy as np
import glob

df=pd.read_csv(r'/home/shivam/Desktop/Datasets/Cancer Dataset/calc_case_description_train_set .csv', usecols = ['patient_id', 'breast density', 'abnormality type', 'calc type', 'calc distribution', 'pathology'])
df=df.replace(np.nan,0)
#print(df.head(10))

#print(df.dtypes)

#print(df["breast density"].value_counts())
#print(df["abnormality type"].value_counts())
#print(df["calc type"].value_counts())
#print(df["calc distribution"].value_counts())
#print(df["pathology"].value_counts())

num_values = {"left or right breast": {"LEFT": 1, "RIGHT": 2},
             "image view": {"MLO":1, "CC": 2},
             "abnormality type": {"calcification":1},
             "calc type": {"PLEOMORPHIC":1, "AMORPHOUS":2, "PUNCTATE":3, "LUCENT_CENTER":4, "VASCULAR":5,"FINE_LINEAR_BRANCHING":6,
                          "COARSE":7, "ROUND_AND_REGULAR-LUCENT_CENTER":8, "PLEOMORPHIC-FINE_LINEAR_BRANCHING":9, "ROUND_AND_REGULAR-LUCENT_CENTER-PUNCTATE":10, 
                          "ROUND_AND_REGULAR-EGGSHELL":11, "PUNCTATE-PLEOMORPHIC":12, "DYSTROPHIC":13, "LUCENT_CENTERED":14, "ROUND_AND_REGULAR":15, "ROUND_AND_REGULAR-LUCENT_CENTERED":16,
                          "AMORPHOUS-PLEOMORPHIC":17,"LARGE_RODLIKE-ROUND_AND_REGULAR":18, "PUNCTATE-AMORPHOUS":19, "COARSE-ROUND_AND_REGULAR-LUCENT_CENTER":20, "VASCULAR-COARSE-LUCENT_CENTERED":21, 
                          "LUCENT_CENTER-PUNCTATE":22, "EGGSHELL":23, "ROUND_AND_REGULAR-PLEOMORPHIC":24, "PUNCTATE-FINE_LINEAR_BRANCHING":25, "VASCULAR-COARSE":26, "ROUND_AND_REGULAR-PUNCTATE":27, "PUNCTATE-ROUND_AND_REGULAR":28,
                          "SKIN-PUNCTATE":29, "SKIN-PUNCTATE-ROUND_AND_REGULAR":30, "LARGE_RODLIKE":31, "COARSE-ROUND_AND_REGULAR-LUCENT_CENTERED":32, "AMORPHOUS-ROUND_AND_REGULAR":33,
                          "PUNCTATE-LUCENT_CENTER":34, "MILK_OF_CALCIUM":35, "SKIN":36, "COARSE-PLEOMORPHIC":37, "ROUND_AND_REGULAR-LUCENT_CENTER-DYSTROPHIC":38, "ROUND_AND_REGULAR-PUNCTATE-AMORPHOUS":39,
                          "VASCULAR-COARSE-LUCENT_CENTER-ROUND_AND_REGULAR-PUNCTATE":40 , "COARSE-ROUND_AND_REGULAR":41,
                          "COARSE-LUCENT_CENTER":42, "PLEOMORPHIC-PLEOMORPHIC":43, "ROUND_AND_REGULAR-AMORPHOUS":44, "SKIN-COARSE-ROUND_AND_REGULAR":45},
             "calc distribution": {"CLUSTERED":1, "SEGMENTAL":2, "REGIONAL":3, "LINEAR":4, "DIFFUSELY_SCATTERED":5, "CLUSTERED-LINEAR":6, "LINEAR-SEGMENTAL":7, "CLUSTERED-SEGMENTAL":8, "REGIONAL-REGIONAL":9},
             "pathology": {"MALIGNANT":1, "BENIGN":2, "BENIGN_WITHOUT_CALLBACK":3}}


df.replace(num_values, inplace=True)
#print(df.head(10))


images_path = []    
path = r'/home/shivam/Desktop/Datasets/Cancer Dataset/Calc ROI Train Images'
for i in os.listdir(path):
    images_path=glob.glob(os.path.join(path,"*","*","*","*.png"))
#df.insert(6,"Images_path")
#print(len(images_path))

df.insert(6,"Images_path",images_path)
df.to_csv("Calci_train_csv.csv",index=False)
print(df.head(5))