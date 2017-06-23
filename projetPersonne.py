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
from sklearn.utils import shuffle
from skimage import io, util, color, feature, transform
import os, os.path
from PIL import ImageDraw
from PIL import Image

#Paths

path="C:\\Users\\Bastien\\Desktop\\projetPers"
pathTrain = path+"\\train"
pathTest = path+"\\test"

#Parameters
heightWindow = 160
widthWindow = 70
nbStepW = 7
nbStepH = 9
nbWindows = 2
rescales = np.arange(0.5,2,0.5)
rescalesTrain = np.arange(3/6,1.5,1/6)
rescalesTest = np.arange(1/6,1.5,1/6)

#Hog params
orientation = 8
pixelPerCell= (8,8)
cellsPerBlock = (1,1)
#clfParams
CTrain = 5
CTest = 5
validCroiseeFolds = 5
seuilDetect = 0.7
seuilDetectTest = 0.5

#Extrait une sous image à partir de l'image passé en paramètre
def extractWindow(im,x,y):
    window = im[y:(y+heightWindow),x:(x+widthWindow)]
    window = feature.hog(window, orientations=orientation, pixels_per_cell=pixelPerCell,cells_per_block=cellsPerBlock, visualise=True)[1]
    window = window.reshape(widthWindow*heightWindow)
    return window

def valid_crois(x,y,clf,nb):
    seuil = math.floor(len(x)/nb)
    res = np.zeros(nb);
    for i in np.arange(1,nb+1): 
        xtest = x[(i-1)*seuil:i*seuil]
        xtrain = np.delete(x,np.arange((i-1)*seuil+1,i*seuil+1),axis=0)
        ytest = y[(i-1)*seuil:i*seuil]
        ytrain = np.delete(y,np.arange((i-1)*seuil+1,i*seuil+1),axis=0)
        clf.fit(xtrain,ytrain)
        taux = np.mean(clf.predict(xtest) != ytest)
        res[i-1]=taux
    return res  
    
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
label = np.concatenate((-1*np.ones((len(files)*nbWindows*len(rescales))),1*np.ones(len(files))),axis=0)
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
    window = feature.hog(window, orientations=orientation, pixels_per_cell=pixelPerCell,cells_per_block=cellsPerBlock, visualise=True)[1]
    windows[j+len(files)*nbWindows*len(rescales)] = window.reshape(widthWindow*heightWindow)
    j=j+1

print("First windows done")
#On ajoute les exemples positifs
print("Starting first training")
#Entrainement du classifieur 
clf = svm.SVC(kernel='linear', probability=True,C=CTrain)
n_samples = windows.shape[0]
#cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
#cross_val_score(clf, windows, label, cv=cv)
windows, label = shuffle(windows,label, random_state=0)
taux = valid_crois(windows,label,clf,validCroiseeFolds)
#clf.fit(windows,label)
fakeWindows = []

print("First Training Done")
print("Error rate first training :")
print(taux)
i = 0
j = 0
print("Starting false positives detection")
#Prédictions négatives sur les images de test
for f in files:
    img = util.img_as_float(color.rgb2gray(io.imread(pathTrain + "\\" + f)))
    currLabel = vect[j]
    print(f)
    for rc in rescalesTrain:
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
                predict = clf.predict_proba(window)[0,1]
                if predict > seuilDetect:
                    labelRescaled = rc * currLabel[1:5]
                    airCommun = abs(X+widthWindow-labelRescaled[0]) * abs(Y+heightWindow-labelRescaled[1])
                    airLabel = labelRescaled[2] * labelRescaled[3]
                    
                    if(airCommun <= 0.6 * airLabel):
                        fakeWindows.append(window[0])
                        #filenum = f.split('.')[0]
                        #results.append([filenum,math.floor(X/rc),math.floor(Y/rc),math.floor(widthWindow/rc),math.floor(heightWindow/rc)])
    j=j+1
     
fakeWindows = np.asarray(fakeWindows)
windows = np.concatenate((windows,fakeWindows))
labelFake = -1*np.ones((len(fakeWindows)))
label = np.concatenate((label,labelFake))
print("False positives added")
print("Training new classifier")

clf = svm.SVC(kernel='linear', probability=True,C=CTest)
n_samples = windows.shape[0]
#cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
#cross_val_score(clf, windows, label, cv=cv)
windows, label = shuffle(windows,label, random_state=0)
taux2 = valid_crois(windows,label,clf,validCroiseeFolds)
#clf.fit(windows,label)
results = []


print("Training done")
print("Error rate second training :")
print(taux2)
print("Starting detection on test images")

#Prédictions positives sur les images de test
