# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 13:56:12 2020

@author: yuezh
"""
#%%

import os
import pandas as pd
import numpy as np

train_df_source = pd.read_csv(os.path.join("CrisisMMD_v2.0", "crisismmd_datasplit_all", "task_humanitarian_text_img_train.tsv"), sep='\t')

# only keep the ones that image & text label align
train_df_aligned = train_df_source[train_df_source.label_text_image=="Positive"]
train_df_aligned.reset_index(drop=True, inplace=True)

train_df_aligned['anomaly'] = 0
train_df_aligned['anomaly'][train_df_aligned['label']=="infrastructure_and_utility_damage"]=1

train_df_aligned.to_excel("train_cleaned.xlsx", index=False)

#%%
test_df_source = pd.read_csv(os.path.join("CrisisMMD_v2.0", "crisismmd_datasplit_all", "task_humanitarian_text_img_test.tsv"), sep='\t')

# only keep the ones that image & text label align
test_df_aligned = test_df_source[test_df_source.label_text_image=="Positive"]
test_df_aligned.reset_index(drop=True, inplace=True)

test_df_aligned['anomaly'] = 0
test_df_aligned['anomaly'][test_df_aligned['label']=="infrastructure_and_utility_damage"]=1

test_df_aligned.to_excel("test_cleaned.xlsx", index=False)



