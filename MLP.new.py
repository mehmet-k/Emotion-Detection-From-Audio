import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.feature_selection as skl
import pandas as pd
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

import util.CreateDataFrames as CDF
import util.CreateTargetVectors as CTV
import util.printScores as PS

K=1
masterDF = CDF.createMasterDataFrameAllEmotions("11","english","MFCC")
#alternative call to above :  CDF.createMasterDataFrame("11","english","MFCC",True,True,True,True,True)
inital_length = len(masterDF)/5
print(inital_length)
for i in range(12,16):
    K+=1
    masterDF = np.concatenate((masterDF,CDF.createMasterDataFrameAllEmotions(str(i),"english","MFCC")))

y=CTV.createTargetVectorALL(inital_length)
for i in range(12,16):
    y = np.concatenate((y, CTV.createTargetVectorALL(inital_length)))

#masterDF.to_csv("master",index=False)
print(masterDF.shape,len(y))

X = masterDF
#X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)
model = MLPClassifier(alpha = 0.01, batch_size = 256, epsilon = 1e-08, hidden_layer_sizes = (300,), learning_rate = 'adaptive', max_iter = 500)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y=CTV.createTargetVectorALL(inital_length)
PS.printScores(y,y_pred)


