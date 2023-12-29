import numpy as np
from sklearn import svm
import util.CreateDataFrames as CDF
import util.CreateTargetVectors as CTV
import util.printScores as PS
from sklearn.model_selection import train_test_split


K=1
masterDF = CDF.createMasterDataFrameAllEmotions("11","english","MEL")
#masterDF = CDF.createMasterDataFrameAllEmotions("11","english","MFCC")

#alternative call to above :  CDF.createMasterDataFrame("11","english","MFCC",True,True,True,True,True)
inital_length = len(masterDF)/5
print(inital_length)
for i in range(12,16):
    K+=1
    masterDF = np.concatenate((masterDF,CDF.createMasterDataFrameAllEmotions(str(i),"english","MEL")))

y=CTV.createTargetVectorALL(inital_length)
X = masterDF
for i in range(12,16):
    y = np.concatenate((y, CTV.createTargetVectorALL(inital_length)))

#masterDF.to_csv("master",index=False)
print(masterDF.shape,len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)

clf = svm.SVC(decision_function_shape='ovo', kernel= "rbf")

clf.fit(X_train,y_train)

# masterDF = CDF.createMasterDataFrameAllEmotions("18","english","MFCC")
# print(masterDF.shape)

df = clf.predict(masterDF)
y=CTV.createTargetVectorALL(inital_length)

PS.printScores(y,df)



