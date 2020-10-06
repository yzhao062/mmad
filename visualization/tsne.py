# -*- coding: utf-8 -*-

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt 
from matplotlib import cm
import matplotlib.patheffects as PathEffects
import seaborn as sns
import pandas as pd

image_emb = np.load('../unimodality/image/train_image_embedding.npy')
text_w2v_emb = np.load('../unimodality/language/word2vec.npy')
text_bert_emb = np.load('../unimodality/language/bert.npy')

image_tsne = TSNE(n_components=2).fit_transform(image_emb)
text_w2v_tsne = TSNE(n_components=2).fit_transform(text_w2v_emb)
text_bert_tsne = TSNE(n_components=2).fit_transform(text_bert_emb)

train_df = pd.read_excel("../train_cleaned.xlsx")
label_image = train_df["label_image"].values
label_text = train_df["label_text"].values

def scatter(x, label):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 8))
    
    # Convert labels from categorical to int, label_name[i]-->i
    label_name, label_idx = np.unique(label, return_inverse=True)
    
    print(label_name)

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[label_idx], alpha=0.7)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(8):
        # Position of each label.
        xtext, ytext = np.median(x[label_idx == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

scatter(image_tsne, label_image)
plt.savefig('image_tsn.png', dpi=120)

scatter(text_w2v_tsne, label_text)
plt.savefig('text_w2v_tsne.png', dpi=120)

scatter(text_bert_tsne, label_text)
plt.savefig('text_bert_tsne.png', dpi=120)