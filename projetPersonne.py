# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import random
from sklearn import svm
from skimage import io, util, color, feature, transform
import os, os.path
from PIL import ImageDraw
from PIL import Image


path="D:\\Bureau\\projetPers"
pathTrain = path+"\\train"
pathTest = path+"\\test"
heightWindow = 140
widthWindow = 60
nbStepW = 10
nbStepH = 10
nbWindows = 5
rescales = np.arange(0.6,2,0.2)

def extractWindow(im,x,y):
    window = im[y:(y+heightWindow),x:(x+widthWindow)]
    window = window.reshape(widthWindow*heightWindow)
    return window
    
    
files = os.listdir(pathTrain)
filesTest = os.listdir(pathTest)
vect = np.zeros((len(files),5))
i=0
with open(path+'\\label.txt') as f:
   for l in f:
       vect[i,:] = l.strip().split(" ")
       i=i+1
vect = vect.astype(int)
n = len(files)*nbWindows*len(rescales)+len(files)
windows = np.zeros((n,heightWindow*widthWindow))
label = np.concatenate((-1*np.ones((len(files)*nbWindows*len(rescales))),np.ones(len(files))),axis=0)
i = 0
#On créé et stocke des fenetres de taille différente aléatoires pour faire des exemples négatifs
for f in files:
    img = util.img_as_float(color.rgb2gray(io.imread(pathTrain + "\\" + f)))  
    for rc in rescales:
        size = (math.floor(img.shape[0]*rc), math.floor(img.shape[1]*rc))
        imgrs = transform.resize(img,size,mode='constant',order=0)
        for p in np.arange(0,nbWindows):
           X = math.floor(random.random()*(imgrs.shape[1]-widthWindow))
           Y = math.floor(random.random()*(imgrs.shape[0]-heightWindow))
           windows[i] = extractWindow(imgrs,X,Y)
           i = i +1
j=0
print("Fake windows done")
#On ajoute les exemples positifs
for f in files:
    img = util.img_as_float(color.rgb2gray(io.imread(pathTrain + "\\" + f)))
    currLabel = vect[j]
    window = img[currLabel[2]:currLabel[2]+currLabel[4],currLabel[1]:currLabel[1]+currLabel[3]]
    window = transform.resize(window,(heightWindow,widthWindow),mode='constant' ,order=0)
    windows[i] = window.reshape(widthWindow*heightWindow)
    i=i+1
    j=j+1
print("Real windows done")
#Entrainement du classifieur 
clf = svm.SVC(kernel='linear', C=1)
clf.fit(windows,label)
results = []
print("Training Done")
#Prédictions sur les images de test
for f in filesTest:
    img = util.img_as_float(color.rgb2gray(io.imread(pathTest + "\\" + f)))
    print(f)
    for rc in rescales:
        size = (math.floor(img.shape[0]*rc), math.floor(img.shape[1]*rc))
        imgrs = transform.resize(img,size,mode='constant',order=0)
        for X in np.arange(0,imgrs.shape[1]-widthWindow,math.floor((imgrs.shape[1]-widthWindow)/nbStepW)):
            for Y in np.arange(0,imgrs.shape[0]-heightWindow,math.floor((imgrs.shape[0]-heightWindow)/nbStepH)):
                window = [extractWindow(imgrs,X,Y)]
                predict = clf.predict(window)
                if predict == 1:
                    filenum = f.split('.')[0]
                    results.append([filenum,math.floor(X/rc),math.floor(Y/rc),math.floor(widthWindow/rc),math.floor(heightWindow/rc)])
                
