import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.feature_selection as skl
import pandas as pd
from sklearn import svm
import util.CreateDataFrames as CDF
import util.CreateTargetVectors as CTV
import util.printScores as PS

K=1
masterDF = CDF.createMasterDataFrameMFCC("11","english",True,True,True,True,True)
inital_length = len(masterDF)/5
print(inital_length)
for i in range(12,17):
    K+=1
    masterDF = np.concatenate((masterDF,CDF.createMasterDataFrameMFCC(str(i),"english",True,True,True,True,True)))

y=CTV.createTargetVectorALL(inital_length)
for i in range(12,17):
    y = np.concatenate((y, CTV.createTargetVectorALL(inital_length)))

#masterDF.to_csv("master",index=False)
print(masterDF.shape,len(y))

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(masterDF,y)

masterDF = CDF.createMasterDataFrameMFCC("18","english",True,True,True,True,True)
print(masterDF.shape)

df = clf.predict(masterDF)
y=CTV.createTargetVectorALL(inital_length)

PS.printScores(y,df)

