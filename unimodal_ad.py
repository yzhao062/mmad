# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 14:35:10 2020

@author: yuezh
"""
import os
import pandas as pd
import numpy as np

from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.copod import COPOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from pyod.utils.utility import standardizer

from sklearn.utils.validation import column_or_1d, check_consistent_length
from sklearn.metrics import roc_auc_score, average_precision_score
from pyod.utils.utility import precision_n_scores


def evaluate_print(clf_name, y, y_pred):
    """Utility function for evaluating and printing the results for examples.
    Default metrics include ROC and Precision @ n

    Parameters
    ----------
    clf_name : str
        The name of the detector.

    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    """

    y = column_or_1d(y)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y, y_pred)

    print('{clf_name} ROC:{roc}, precision @ rank n:{prn}, ap:{prn}'.format(
        clf_name=clf_name,
        roc=np.round(roc_auc_score(y, y_pred), decimals=4),
        prn=np.round(precision_n_scores(y, y_pred), decimals=4),
        ap=np.round(average_precision_score(y, y_pred), decimals=4)))
    
# first read in anomaly labels

train_df = pd.read_excel("train_cleaned.xlsx")
anomaly_label = train_df["anomaly"]

# read in unimodal representation
unimodal_embeddings = [
    np.load(os.path.join("unimodality", "image", "train_image_embedding.npy")),
    np.load(os.path.join("unimodality", "language", "bert.npy")),
    np.load(os.path.join("unimodality", "language", "word2vec.npy")),
    np.load(os.path.join("multimodality", "baseline_wen_embeding.npy")),
    np.load(os.path.join("multimodality", "vae_joint_representation.npy")),
    np.concatenate([np.load(os.path.join("unimodality", "image", "train_image_embedding.npy")), 
                    np.load(os.path.join("unimodality", "language", "word2vec.npy"))], axis=1)
]
unimodality = ["image", "word2vec", "bert", "concat_joint", "vae_joint", "simple_concat"]


clfs = [IForest(random_state=42), LOF(), OCSVM(), PCA(), KNN(), HBOS(), COPOD(), 
        AutoEncoder(verbose=0), VAE(latent_dim=32, verbosity=0)]

for embedding, modality in zip(unimodal_embeddings, unimodality):
    print()
    print(modality)
    print()
    embedding_scaled = standardizer(embedding)

    for clf in clfs:
        # print(clf)
        clf.fit(embedding_scaled)
        evaluate_print(clf.__class__.__name__, anomaly_label,
                       clf.decision_scores_)
        
#%%
image_text_embedding = [
    np.load(os.path.join("unimodality", "image", "train_image_embedding.npy")),
    np.load(os.path.join("unimodality", "language", "word2vec.npy")),
    ]

print("score averaging")

for clf in clfs:
    # print(clf)
    clf.fit(image_text_embedding[0])
    temp_score_0 = clf.decision_scores_
    
    clf.fit(image_text_embedding[1])
    temp_score_1 = clf.decision_scores_    
    
    evaluate_print(clf.__class__.__name__, anomaly_label,
                   temp_score_0 + temp_score_1)
        
