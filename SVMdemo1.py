import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.feature_selection as skl
import pandas as pd
from sklearn import svm
import CreateTargetVectors as CTV
import CreateDataFrames as CDF



masterDF = CDF.createMasterDataFrame_ANGRY_HAPPY(str(11))
for i in range(12,16):
    masterDF = np.concatenate((masterDF,CDF.createMasterDataFrame_ANGRY_HAPPY(str(i))))

y = CTV.createTargetVector_ANGRY_HAPPY()
for i in range(12,16):
    y = np.append(y,CTV.createTargetVector_ANGRY_HAPPY())

print(masterDF.shape,len(y))
for i in range(len(y)):
    print(masterDF[i],y[i])


clf = svm.SVC(decision_function_shape='ovo',class_weight={"Angry":7,"Happy":3})
clf.fit(masterDF,y)

#dec = clf.decision_function(y)
testDF = CDF.createMasterDataFrame_ANGRY_HAPPY("16")

print(testDF.shape)

df = clf.predict(testDF)
#lin_clf = svm.LinearSVC(dual="auto")
#lin_clf.fit(masterDF, y)
#dec = lin_clf.decision_function(masterDF)

#df = lin_clf.predict(masterDF)
y = CTV.createTargetVector_ANGRY_HAPPY()
#dfx = pd.DataFrame(df)
#dfx.to_csv("predictions.csv",index=False)
scores=[0,0]
for j in range(0,700):
    if y[j]==df[j] and y[j] == "Happy":
        scores[0] = scores[0]+1
    elif y[j]==df[j] and y[j] == "Angry":
        scores[1] = scores[1]+1

print("happy score:", scores[0]/350*100)

print("angry score:", scores[1]/350*100)
