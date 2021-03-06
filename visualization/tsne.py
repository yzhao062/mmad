# -*- coding: utf-8 -*-

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt 
from matplotlib import cm
import matplotlib.patheffects as PathEffects
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
image_emb = np.load('../unimodality/image/train_image_embedding.npy')
text_w2v_emb = np.load('../unimodality/language/word2vec.npy')
text_bert_emb = np.load('../unimodality/language/bert.npy')

mulmodal_wen_emb = np.load('../multimodality/baseline_wen_embeding.npy')
mulmodal_con_emb = np.load('../multimodality/input_concat.npy')
mulmodal_vae_emb = np.load('../multimodality/vae_joint_representation.npy')

image_tsne = TSNE(n_components=2).fit_transform(StandardScaler().fit_transform(image_emb))
text_w2v_tsne = TSNE(n_components=2).fit_transform(StandardScaler().fit_transform(text_w2v_emb))
text_bert_tsne = TSNE(n_components=2).fit_transform(StandardScaler().fit_transform(text_bert_emb))

mulmodal_wen_tsne = TSNE(n_components=2).fit_transform(StandardScaler().fit_transform(mulmodal_wen_emb))
mulmodal_con_tsne = TSNE(n_components=2).fit_transform(StandardScaler().fit_transform(mulmodal_con_emb))
mulmodal_vae_tsne = TSNE(n_components=2).fit_transform(StandardScaler().fit_transform(mulmodal_vae_emb))

train_df = pd.read_excel("../train_cleaned.xlsx")
label_image = train_df["label_image"].values
label_text = train_df["label_text"].values

label = train_df["anomaly"].values

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
    
    #ls = [mpatches.Patch(color=palette[i],label=str(i)) for i in range(8)]
    ls = [mpatches.Patch(color=palette[i],label=str(i)) for i in range(2)]
    ax.legend(handles=ls)
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

scatter(image_tsne, label)
plt.savefig('image_tsn.png', dpi=120)

scatter(text_w2v_tsne, label)
plt.savefig('text_w2v_tsne.png', dpi=120)

scatter(text_bert_tsne, label)
plt.savefig('text_bert_tsne.png', dpi=120)

scatter(mulmodal_wen_tsne, label)
plt.savefig('mulmodal_wen_tsne.png', dpi=120)

scatter(mulmodal_con_tsne, label)
plt.savefig('mulmodal_con_tsne.png', dpi=120)

scatter(mulmodal_vae_tsne, label)
plt.savefig('mulmodal_vae_tsne.png', dpi=120)