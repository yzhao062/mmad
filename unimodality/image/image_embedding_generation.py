# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 12:18:02 2020

@author: yuezh
"""
import os 
import pandas as pd
import numpy as np
from img2vec_pytorch import Img2Vec
from PIL import Image

# Initialize Img2Vec with GPU
img2vec = Img2Vec(cuda=True)

#%%

# examples 
img = Image.open('917791130590183424_0.jpg')
vec = img2vec.get_vec(img, tensor=True)

images = ['917791130590183424_0.jpg', '917791291823591425_0.jpg']

p_images = []
for i in images:
    p_images.append(Image.open(i))
vectors = img2vec.get_vec(p_images)

#%%

# build image embeddings for train dataset
# this path should be changed
train_df = pd.read_excel("C:\\Users\\yuezh\\PycharmProjects\\mmad\\train_cleaned.xlsx")


# construct image queue
image_loc_list = []

for index, row in train_df.iterrows():
    image_loc_list.append(os.path.join("C:\\Users\\yuezh\\PycharmProjects\\mmad\\CrisisMMD_v2.0", 
                                       row["image"].replace('/', "\\")))
    
# construct PIL image queue

image_PIL_list =[]

for loc in image_loc_list:
    image_PIL_list.append(Image.open(loc).convert('RGB'))
    
#%%
# generate image vectors (use small batches)
# image_embedding = img2vec.get_vec(image_PIL_list[0:1000])

image_mat = np.zeros([len(image_PIL_list), 512])
for i in range(len(image_PIL_list)):
    print("processing", i)
    image_mat[i, :] = img2vec.get_vec(image_PIL_list[i])

np.save('train_image_embedding.npy', image_mat)