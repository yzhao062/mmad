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
from pyod.utils.data import evaluate_print
from pyod.utils.utility import standardizer

# first read in anomaly labels

train_df = pd.read_excel("train_cleaned.xlsx")
anomaly_label = train_df["anomaly"]

# read in unimodal representation
unimodal_embeddings = [
    np.load(os.path.join("../unimodality", "image", "train_image_embedding.npy")),
    np.load(os.path.join("../unimodality", "language", "bert.npy")),
    np.load(os.path.join("../unimodality", "language", "word2vec.npy")),
]
unimodality = ["image", "word2vec", "bert"]

clfs = [IForest(), LOF(), OCSVM(), PCA(), KNN(), HBOS(), COPOD()]

for embedding, modality in zip(unimodal_embeddings, unimodality):
    print(modality)
    embedding_scaled = standardizer(embedding)

    for clf in clfs:
        # print(clf)
        clf.fit(embedding_scaled)
        evaluate_print(clf.__class__.__name__, anomaly_label,
                       clf.decision_scores_)
