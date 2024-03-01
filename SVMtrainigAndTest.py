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
import util.printScores as PS

y=CDV.createTargetVectorALL(350)
print(len(y))
masterDF = CDF.createMasterDataFrameAllEmotions(str(11),"english","MFCC")
print(masterDF.shape)
for i in range (12,16):
    masterDF = np.concatenate((masterDF,CDF.createMasterDataFrameAllEmotions(str(i),"english","MFCC")))
for i in range(12,16):
    y = np.concatenate((y, CDV.createTargetVectorALL(350)))
#masterDF.to_csv("master",index=False)
print(masterDF.shape,len(y))
print(masterDF,y)

X_train, X_test, y_train, y_test = train_test_split(masterDF, y, stratify=y,random_state=1)

lin_clf = svm.SVC(decision_function_shape='ovo', kernel= "rbf")
lin_clf.fit(X_train, y_train)


y_pred = lin_clf.predict(X_test)

PS.printScoresOld(y_pred,y_test)