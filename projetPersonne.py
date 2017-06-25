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
heightWindow = 180
widthWindow = 80
nbStepW = 10
nbStepH = 12
nbStepWTest = 10
nbStepHTest = 12
nbWindows = 4
rescales = np.arange(1/6,8/6,1/6)
rescalesTrain = np.arange(1/6,9/6,1/6)
rescalesTest = np.arange(1/6,7/6,1/6)

#Hog params
orientation = 9
pixelPerCell= (2,2)
cellsPerBlock = (1,1)
#clfParams
CTrain = 8
CTest = 8
validCroiseeFolds = 5


#Extrait une sous image à partir de l'image passé en paramètre
def extractWindow(im,x,y):
    window = im[y:(y+heightWindow),x:(x+widthWindow)]
    window = feature.hog(window, orientations=orientation, pixels_per_cell=pixelPerCell,cells_per_block=cellsPerBlock)
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
    
def writeResults(res):
    file = open("results.txt","w")
    for i in np.arange(0,len(res)):
        num = str(res[i][0])
        x = str(res[i][1])
        y = str(res[i][2])
        w = str(res[i][3])
        h = str(res[i][4])
        prob = str(res[i][5])
        file.write(num+" "+x+" "+y+" "+w+" "+h+" "+prob+"\n")
    file.close()
    
def displayim(pathIm,lab):
    im = Image.open(pathIm)
    imdr = ImageDraw.Draw(im)
    imdr.line([(lab[1],lab[2]),(lab[1]+lab[3],lab[2])], (0,200,255), width=5)
    imdr.line([(lab[1],lab[2]),(lab[1],lab[2]+lab[4])], (0,200,255), width=5)
    imdr.line([(lab[1],lab[2]+lab[4]),(lab[1]+lab[3],lab[2]+lab[4])], (0,200,255), width=5)
    imdr.line([(lab[1]+lab[3],lab[2]),(lab[1]+lab[3],lab[2]+lab[4])], (0,200,255), width=5)
    im.show()


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
    
def non_max_suppression(boxes, overlapThresh):
    boxes = np.asarray(boxes)
    if len(boxes) == 0:
        return []

    
    boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:,1]
    y1 = boxes[:,2]
    x2 = boxes[:,1]+boxes[:,3]
    y2 = boxes[:,2]+boxes[:,4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    retList = []
    for i in pick:
        retList.append([str(int(boxes[i,0])).zfill(3),int(boxes[i,1]),int(boxes[i,2]),int(boxes[i,3]),int(boxes[i,4]),boxes[i,5]])
    return retList
    
def removeRealWindows(detections,correction):
    fakeWin = []
    for i in np.arange(0,len(detections)):
        numFile = int(detections[i][0])
        corr = correction[numFile]
        X1,Y1,W1,H1 = detections[i][1],detections[i][2],detections[i][3],detections[i][4]
        X2,Y2,W2,H2 = corr[1],corr[2],corr[3],corr[4]
        if (X1>X2 and X1+W1<X2+W2 and Y1>Y2 and Y1+H1<Y2+H2):#Si la boite est entièrement contenue dans la vraie (=boite trop petite)
            fakeWin.append(detections[i])
        elif (X1+W1<X2 or X2+W2<X1 or Y1+H1<Y2 or Y2+H2<Y1):#Si il n'y a pas d'intersection
            fakeWin.append(detections[i])
    return fakeWin
    
def isRealWindow(corr,x,y,h,w):
    X1,Y1,W1,H1 = x,y,w,h
    X2,Y2,W2,H2 = corr[1],corr[2],corr[3],corr[4]
    if (X1>X2 and X1+W1<X2+W2 and Y1>Y2 and Y1+H1<Y2+H2):#Si la boite est entièrement contenue dans la vraie (=boite trop petite)
        return 0
    elif (X1+W1<X2 or X2+W2<X1 or Y1+H1<Y2 or Y2+H2<Y1):#Si il n'y a pas d'intersection
        return 0
    return 1
    
    
    
    
    
    
    
    
files = os.listdir(pathTrain)
filesTest = os.listdir(pathTest)
vect = np.zeros((len(files),5))
i=0
with open(path+'\\label.txt') as f:
   for l in f:
       vect[i,:] = l.strip().split(" ")
       i=i+1
vect = vect.astype(int)
n = len(files)*nbWindows*len(rescales)
splitWin = (math.floor(widthWindow/pixelPerCell[0]),math.floor(heightWindow/pixelPerCell[1]))
sizeWindow = splitWin[0]*splitWin[1]*orientation
randomWindows = np.zeros((n,sizeWindow))
trueWindows = np.zeros((len(files),sizeWindow))



#On créé et stocke des fenetres de taille différente aléatoires pour faire des exemples négatifs et positifs
i = 0
j = 0
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
           #On vérifie qu'elles ne sont pas des bonnes detections
           if (isRealWindow(vect[j],math.floor(X/rc),math.floor(Y/rc),math.floor(widthWindow/rc),math.floor(heightWindow/rc)) !=1):
               randomWindows[i] = extractWindow(imgrs,X,Y)
               i = i +1
    currLabel = vect[j]
    window = img[currLabel[2]:currLabel[2]+currLabel[4],currLabel[1]:currLabel[1]+currLabel[3]]
    window = transform.resize(window,(heightWindow,widthWindow),mode='constant' ,order=0)
    window = feature.hog(window, orientations=orientation, pixels_per_cell=pixelPerCell,cells_per_block=cellsPerBlock)
    trueWindows[j]= window
    j=j+1
randomWindows = np.delete(randomWindows,np.arange(i,len(randomWindows)),axis=0)
windows = np.concatenate((randomWindows,trueWindows),axis=0)
label = np.concatenate((-1*np.ones((len(randomWindows))),1*np.ones(len(trueWindows))),axis=0)
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


print("First Training Done")
print("Error rate first training :")
print(taux)
i = 0
j = 0
print("Starting false positives detection")

fakeWindows = []
#Prédictions négatives sur les images de test
for f in files:
    img = util.img_as_float(color.rgb2gray(io.imread(pathTrain + "\\" + f)))
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
                if (isRealWindow(vect[j],math.floor(X/rc),math.floor(Y/rc),math.floor(widthWindow/rc),math.floor(heightWindow/rc)) !=1):
                    window = [extractWindow(imgrs,X,Y)]
                    predict = clf.predict(window)
                    if predict == 1:
                        fakeWindows.append(window[0])
                        i=i+1
                        #filenum = f.split('.')[0]
                        #results.append([filenum,math.floor(X/rc),math.floor(Y/rc),math.floor(widthWindow/rc),math.floor(heightWindow/rc)])
    j=j+1

fakeWindows = np.asarray(fakeWindows)
windows = np.concatenate((windows,fakeWindows))
labelFake = -1*np.ones((len(fakeWindows)))
label = np.concatenate((label,labelFake))
print("False positives added")
print(str(len(fakeWindows))+" false positives found")
print("Training new classifier")

clfTest = svm.SVC(kernel='linear', probability=True,C=CTest)
n_samples = windows.shape[0]
#cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
#cross_val_score(clf, windows, label, cv=cv)
windows, label = shuffle(windows,label, random_state=0)
taux2 = valid_crois(windows,label,clfTest,validCroiseeFolds)



print("Training done")
print("Error rate second training :")
print(taux2)
print("Starting detection on test images")

results = []
#Prédictions positives sur les images de test

    
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





"""
currFile = ""
temp = []


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
    
writeResults(finalresults2)


currFile = ""
temp = []

finalresults = []
for i in np.arange(0,len(results)):
    if(currFile == results[i][0]):
        temp.append(results[i])
    elif(currFile != results[i][0]):
        if(len(temp)==1):
            finalresults.append(temp[0])
        elif(len(temp)!=0):
            addFinal(temp,finalresults)
        temp=[]
        currFile=results[i][0]
        temp.append(results[i])
    if(i==len(results)-1):
        if(len(temp)==1):
            finalresults.append(temp[0])
        elif(len(temp)!=0):
            addFinal(temp,finalresults)
            
for i in np.arange(0,len(finalresults)):
    displayim(pathTest+'\\'+finalresults[i][0]+".jpg",finalresults[i])


writeResults(finalresults)
"""
currFile = ""
temp = []

nmfinalresults = []
for i in np.arange(0,len(results)):
    if(currFile == results[i][0]):
        temp.append(results[i])
    elif(currFile != results[i][0]):
        nm_res = non_max_suppression(temp,0.5)
        for j in np.arange(0,len(nm_res)):
            nmfinalresults.append(nm_res[j])
        temp=[]
        currFile=results[i][0]
        temp.append(results[i])
    if(i==len(results)-1):
        nm_res = non_max_suppression(temp,0.5)
        for j in np.arange(0,len(nm_res)):
            nmfinalresults.append(nm_res[j])
            
for i in np.arange(0,len(nmfinalresults)):
    displayim(pathTest+'\\'+nmfinalresults[i][0]+".jpg",nmfinalresults[i])


writeResults(nmfinalresults)
