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
nbStepW = 8
nbStepH = 10
nbStepWTest = 6
nbStepHTest = 7
nbWindows = 2
rescales = np.arange(1/6,8/6,1/6)
rescalesTrain = np.arange(1/6,1,1/6)
rescalesTest = np.arange(1/6,1,1/6)

#Hog params
orientation = 8
pixelPerCell= (10,10)
cellsPerBlock = (1,1)
#clfParams
CTrain = 8
CTest = 8
validCroiseeFolds = 5
seuilDetect = 0.6
seuilDetectTest = 0.4

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
def aRecouv(bi,bj,seuil):
    maxgauche=max(bi[1],bj[1])
    mindroit=min(bi[1]+bi[3],bj[1]+bj[3])
    minhaut=max(bi[2],bj[2])
    maxbas=min(bi[2]+bi[4],bj[2]+bj[4])
    if(maxgauche<mindroit and minhaut<maxbas):
        intersect = (mindroit-maxgauche)*(maxbas-minhaut)
        union = (bi[3]*bi[4])+(bj[3]*bj[4])-intersect
        if(intersect/union>seuil):
            return 1
        else:
            return 0
    else:
        return 0
    
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
testFalse = []
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
        stepX = math.floor((imgrs.shape[0]-heightWindow)/nbStepW)
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
                    labelRescaled = rc * currLabel
                    airCommun = abs(X+widthWindow-labelRescaled[0]) * abs(Y+heightWindow-labelRescaled[1])
                    label2 = [f.split('.')[0],X,Y,widthWindow,heightWindow]
                    if(aRecouv(label2,labelRescaled,0.5) == 0):
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

clfTest = svm.SVC(kernel='linear', probability=True,C=CTest)
n_samples = windows.shape[0]
#cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
#cross_val_score(clf, windows, label, cv=cv)
windows, label = shuffle(windows,label, random_state=0)
taux2 = valid_crois(windows,label,clfTest,validCroiseeFolds)
clf.fit(windows,label)



print("Training done")
print("Error rate second training :")
print(taux2)
print("Starting detection on test images")

#Prédictions positives sur les images de test
results = []
for f in filesTest:
    img = util.img_as_float(color.rgb2gray(io.imread(pathTest + "\\" + f)))
    print(f)
    for rc in rescalesTest:
        size = (math.floor(img.shape[0]*rc), math.floor(img.shape[1]*rc))
        if(size[0]<heightWindow or size[1]<widthWindow):
            size=(heightWindow,widthWindow)
        imgrs = transform.resize(img,size,mode='constant',order=0)
        stepX = math.floor((imgrs.shape[0]-heightWindow)/nbStepWTest)
        if (stepX==0):
            stepX=1
        for X in np.arange(0,imgrs.shape[1]-widthWindow,stepX):
            stepY = math.floor((imgrs.shape[0]-heightWindow)/nbStepHTest)
            if (stepY==0):
                stepY=1
            for Y in np.arange(0,imgrs.shape[0]-heightWindow,stepY):
                window = [extractWindow(imgrs,X,Y)]
                predict = clfTest.predict_proba(window)[0,1]
                pred = clfTest.predict(window)
                if pred == 1:
                    filenum = f.split('.')[0]
                    results.append([filenum,math.floor(X/rc),math.floor(Y/rc),math.floor(widthWindow/rc),math.floor(heightWindow/rc),predict])

print("Detection done !")



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

def addFinal(temp,final):
    temp = sorted(temp,reverse=True,key=lambda x: x[5])
    k = 1
    for j in np.arange(1,len(temp)):
        if(aRecouv(temp[j-k],temp[j],0.5) == 0):
            final.append(temp[j-k])
            k=1
        if(aRecouv(temp[j-k],temp[j],0.5) == 1):
            k=k+1
def addFinal2(temp,final):
    temp = sorted(temp,reverse=True,key=lambda x: x[5])
    final.append(temp[0])
    
currFile = ""
temp = []
finalresults = []
finalresults2 = []
for i in np.arange(0,len(results)):
    if(currFile == results[i][0]):
        temp.append(results[i])
    elif(currFile != results[i][0]):
        if(len(temp)==1):
            finalresults2.append(temp[0])
        elif(len(temp)!=0):
            addFinal2(temp,finalresults2)
        temp=[]
        currFile=results[i][0]
        temp.append(results[i])
    if(i==len(results)-1):
        if(len(temp)==1):
            finalresults2.append(temp[0])
        elif(len(temp)!=0):
            addFinal2(temp,finalresults2)

for i in np.arange(0,len(finalresults2)):
    displayim(pathTest+'\\'+finalresults2[i][0]+".jpg",finalresults2[i])
