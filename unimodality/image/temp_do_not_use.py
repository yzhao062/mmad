# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 12:18:02 2020

@author: yuezh
"""
from img2vec_pytorch import Img2Vec
from PIL import Image

# Initialize Img2Vec with GPU
img2vec = Img2Vec(cuda=True)

#%%
img = Image.open('917791130590183424_0.jpg')
vec = img2vec.get_vec(img, tensor=True)

images = ['917791130590183424_0.jpg', '917791291823591425_0.jpg']

p_images = []
for i in images:
    p_images.append(Image.open(i))
vectors = img2vec.get_vec(p_images)
