import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.feature_selection as skl
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

import util.CreateDataFrames as CDF
import util.CreateTargetVectors as CDV



y=CDV.createTargetVectorALL(349)
print(len(y))
masterDF = CDF.createMasterDataFrameAllEmotions(str(11),"english","MFCC")
print(masterDF.shape)
for i in range (12,16):
    masterDF = np.concatenate((masterDF,CDF.createMasterDataFrameAllEmotions(str(i),"english","MFCC")))
for i in range(12,16):
    y = np.concatenate((y, CDV.createTargetVectorALL(349)))
#masterDF.to_csv("master",index=False)
print(masterDF.shape,len(y))
print(masterDF,y)

X_train, X_test, y_train, y_test = train_test_split(masterDF, y, stratify=y,random_state=1)

lin_clf = svm.SVC(decision_function_shape='ovo', kernel= "rbf")
lin_clf.fit(X_train, y_train)


y_pred = lin_clf.predict(X_test)
scores=[0,0,0,0,0]
print(len(y_pred),y_pred)
for j in range(0,len(y_pred)):
    if y_test[j]==y_pred[j] and y_test[j] == "Angry":
        scores[0] = scores[0]+1
    elif y_test[j]==y_pred[j] and y_test[j] == "Happy":
        scores[1] = scores[1]+1
    elif y_test[j]==y_pred[j] and y_test[j] == "Neutral":
        scores[2] = scores[2]+1
    elif y_test[j]==y_pred[j] and y_test[j] == "Sad":
        scores[3] = scores[3]+1
    elif y_test[j]==y_pred[j] and y_test[j] == "Surprise":
        scores[4] = scores[4]+1
count = [0,0,0,0,0]
for j in range(0,len(y_pred)):
    if y_test[j]== "Angry":
        count[0] += 1
    elif y_test[j]== "Happy":
        count[1] += 1
    elif y_test[j]== "Neutral":
        count[2] += 1
    elif y_test[j]== "Sad":
        count[3] += 1
    elif y_test[j]== "Surprise":
        count[4] += 1
for i in range(0,len(count)):
    print(count[i])

print("angry score:",scores[0]/count[0]*100)

print("happy score:", scores[1]/count[1]*100)

print("neutral score:", scores[2]/count[2]*100)

print("sad score:", scores[3]/count[3]*100)

print("surprise score:", scores[4]/count[4]*100)
