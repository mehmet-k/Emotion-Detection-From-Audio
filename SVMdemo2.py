import numpy as np
from sklearn import svm
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

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(masterDF,y)

# masterDF = CDF.createMasterDataFrameAllEmotions("18","english","MFCC")
# print(masterDF.shape)

df = clf.predict(masterDF)
y=CTV.createTargetVectorALL(inital_length)

PS.printScores(y,df)



