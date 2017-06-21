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
stepW = 25
stepH = 40
nbWindows = 5
rescales = np.arange(0.6,2,0.2)

def extractWindow(im,x,y):
    window = im[y:(y+heightWindow),x:(x+widthWindow)]
    window = window.reshape(widthWindow*heightWindow)
    return window
    
    
files = os.listdir(pathTrain)
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
print("False windows stored")
for f in files:
    img = util.img_as_float(color.rgb2gray(io.imread(pathTrain + "\\" + f)))
    currLabel = vect[j]
    window = img[currLabel[2]:currLabel[2]+currLabel[4],currLabel[1]:currLabel[1]+currLabel[3]]
    window = transform.resize(window,(heightWindow,widthWindow),mode='constant' ,order=0)
    windows[i] = window.reshape(widthWindow*heightWindow)
    i=i+1
    j=j+1
  
#Entrainement du classifieur 
clf = svm.SVC(kernel='linear', C=1)
clf.fit(windows,label)
results = []

#Prédictions sur les images de test
for f in filesTest:
    img = util.img_as_float(color.rgb2gray(io.imread(pathTest + "\\" + f)))
    for rc in rescales:
        size = (math.floor(img.shape[0]*rc), math.floor(img.shape[1]*rc))
        imgrs = transform.resize(img,size,mode='constant',order=0)
        for X in np.arange(0,img.shape[1]-widthWindow,stepW):
            for Y in np.arange(0,img.shape[0]-heightWindow,stepH):
                window = extractWindow(imgrs,X,Y)
                predict = clf.predict(window)
                if predict == 1:
                    filenum = f.split('.')[0]
                    results.append([filenum,math.floor(X/rc),math.floor(Y/rc,math.floor(widthWindow/rc),math.floor(heightWindow/rc))])


"""
vect = np.zeros((len(imgs),5))
i=0
with open(path+'\\label.txt') as f:
   for l in f:
       vect[i,:] = l.strip().split(" ")
       i=i+1
dimBoite = np.zeros((140,60))


im = Image.open(pathTrain+"\\005.jpg")
imdr = ImageDraw.Draw(im)
lab = vect[4]
imdr.line([(lab[1],lab[2]),(lab[1]+lab[3],lab[2])], (0,200,255), width=5)
imdr.line([(lab[1],lab[2]),(lab[1],lab[2]+lab[4])], (0,200,255), width=5)
imdr.line([(lab[1],lab[2]+lab[4]),(lab[1]+lab[3],lab[2]+lab[4])], (0,200,255), width=5)
imdr.line([(lab[1]+lab[3],lab[2]),(lab[1]+lab[3],lab[2]+lab[4])], (0,200,255), width=5)
im.show()
imgsArr = np.array(np.zeros(len(imgs)),dtype=object)
for i in np.arange(len(imgs)-1,0,-1):
    img = imgs.pop();
    imgsArr[i] = img.reshape(img.shape[0]*img.shape[1])

clf = svm.SVC(kernel='linear',C=15)
clf.fit(imgsArr,vect[:,1:5])
clf = svm.SVC(kernel='linear',C=15)
for i in np.arange(len(imgs)-1,0):
    label = vect[i]
    im = imgs.pop();
    clf.fit(im.reshape(im.shape[0]*im.shape[1]),label[1:4])
s = clf.predict(imgsTest.pop())
    
"""
