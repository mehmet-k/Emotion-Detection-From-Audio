import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.feature_selection as skl
import pandas as pd
from sklearn import svm
import util.CreateDataFrames as CDF
import util.CreateTargetVectors as CDV



y=[]
masterDF = [[]]
for i in range (11,16):
    masterDF = CDF.createMasterDataFrameAllEmotions(str(i),"english","MFCC")
for i in range(11,16):
    y = np.concatenate((y, CDV.createTargetVectorALL(350)))
#masterDF.to_csv("master",index=False)
print(masterDF.shape,len(y))
print(masterDF,y)

lin_clf = svm.SVC(decision_function_shape='ovo',kernel="rbf")
lin_clf.fit(masterDF, y)
dec = lin_clf.decision_function(masterDF)


masterDF = createMasterDataFrameMFCC("15")
print(masterDF.shape)
df = lin_clf.predict(masterDF)

y = createTargetVector()
scores=[0,0,0,0,0]
