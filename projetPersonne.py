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
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from skimage import io, util, color, feature, transform
import os, os.path
from PIL import ImageDraw
from PIL import Image

#Paths

path="C:\\Users\\Bastien\\Desktop\\projetPers"
pathTrain = path+"\\train"
pathTest = path+"\\test"

#Parameters
heightWindow = 140
widthWindow = 60
nbStepW = 8
nbStepH = 12
nbWindows = 4
rescales = np.arange(1/6,8/6,1/6)
rescalesTest = np.arange(1/6,8/6,1/6)


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
j = 0
#On créé et stocke des fenetres de taille différente aléatoires pour faire des exemples négatifs et positifs
for f in files:
    img = util.img_as_float(color.rgb2gray(io.imread(pathTrain + "\\" + f)))  
    for rc in rescales:
        size = (math.floor(img.shape[0]*rc), math.floor(img.shape[1]*rc))
        if(size[0]<heightWindow or size[1]<widthWindow):
            size=(heightWindow,widthWindow)
        imgrs = transform.resize(img,size,mode='constant',order=0)
        for p in np.arange(0,nbWindows):
           X = math.floor(random.random()*(imgrs.shape[1]-widthWindow))
           Y = math.floor(random.random()*(imgrs.shape[0]-heightWindow))
           windows[i] = extractWindow(imgrs,X,Y)
           i = i +1
    currLabel = vect[j]
    window = img[currLabel[2]:currLabel[2]+currLabel[4],currLabel[1]:currLabel[1]+currLabel[3]]
    window = transform.resize(window,(heightWindow,widthWindow),mode='constant' ,order=0)
    windows[j+len(files)*nbWindows*len(rescales)] = window.reshape(widthWindow*heightWindow)
    j=j+1

print("Fake windows done")
#On ajoute les exemples positifs

#Entrainement du classifieur 
clf = svm.SVC(kernel='linear', C=1)
n_samples = windows.shape[0]
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
cross_val_score(clf, windows, label, cv=cv)
clf.fit(windows,label)
results = []

print("Training Done")
#Prédictions sur les images de test
for f in filesTest:
    img = util.img_as_float(color.rgb2gray(io.imread(pathTest + "\\" + f)))
    print(f)
    for rc in rescalesTest:
        size = (math.floor(img.shape[0]*rc), math.floor(img.shape[1]*rc))
        if(size[0]<heightWindow or size[1]<widthWindow):
            size=(heightWindow,widthWindow)
        imgrs = transform.resize(img,size,mode='constant',order=0)
        stepX = math.floor((imgrs.shape[0]-heightWindow)/nbStepH)
        if (stepX==0):
            stepX=1
        for X in np.arange(0,imgrs.shape[1]-widthWindow,stepX):
            stepY = math.floor((imgrs.shape[0]-heightWindow)/nbStepH)
            if (stepY==0):
                stepY=1
            for Y in np.arange(0,imgrs.shape[0]-heightWindow,stepY):
                window = [extractWindow(imgrs,X,Y)]
                predict = clf.predict(window)
                if predict == 1:
                    filenum = f.split('.')[0]
                    results.append([filenum,math.floor(X/rc),math.floor(Y/rc),math.floor(widthWindow/rc),math.floor(heightWindow/rc)])
def displayim(pathIm,lab):
    im = Image.open(pathIm)
    imdr = ImageDraw.Draw(im)
    imdr.line([(lab[1],lab[2]),(lab[1]+lab[3],lab[2])], (0,200,255), width=5)
    imdr.line([(lab[1],lab[2]),(lab[1],lab[2]+lab[4])], (0,200,255), width=5)
    imdr.line([(lab[1],lab[2]+lab[4]),(lab[1]+lab[3],lab[2]+lab[4])], (0,200,255), width=5)
    imdr.line([(lab[1]+lab[3],lab[2]),(lab[1]+lab[3],lab[2]+lab[4])], (0,200,255), width=5)
    im.show()
"""
for i in np.arange(0,len(results)):
    displayim(pathTest+'\\'+results[i][0]+".jpg",results[i])
"""
