import numpy as np
from sklearn import svm
import util.CreateDataFrames as CDF
import util.CreateTargetVectors as CTV
import util.printScores as PS
from sklearn.model_selection import train_test_split


K=1
masterDF = CDF.createMasterDataFrameAllEmotions("11","english","MFCC")
#masterDF = CDF.createMasterDataFrameAllEmotions("11","english","MFCC")

#alternative call to above :  CDF.createMasterDataFrame("11","english","MFCC",True,True,True,True,True)
inital_length = len(masterDF)/5
print(inital_length)
for i in range(12,16):
    K+=1
    masterDF = np.concatenate((masterDF,CDF.createMasterDataFrameAllEmotions(str(i),"english","MFCC")))

y=CTV.createTargetVectorALL(inital_length)
X = masterDF
for i in range(12,16):
    y = np.concatenate((y, CTV.createTargetVectorALL(inital_length)))

#masterDF.to_csv("master",index=False)
print(masterDF.shape,len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Don't cheat - fit only on training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# apply same transformation to test data
X_test = scaler.transform(X_test)

clf = svm.SVC(decision_function_shape='ovo', kernel= "rbf")

clf.fit(X_train,y_train)

# masterDF = CDF.createMasterDataFrameAllEmotions("18","english","MFCC")
# print(masterDF.shape)

df = clf.predict(X_test)
y=CTV.createTargetVectorALL(inital_length)

PS.printScores(y_test,df)



