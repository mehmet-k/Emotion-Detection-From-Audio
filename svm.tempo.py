import numpy as np
from sklearn import svm
import util.CreateDataFrames as CDF
import util.CreateTargetVectors as CTV
import util.printScores as PS
from sklearn.model_selection import train_test_split

K = 1
masterDF = CDF.createMasterDataFrameAllEmotions("11", "english", "TEMPO")
initial_length = len(masterDF) // 5
print(initial_length)

# Initialize empty lists for X and y
X_list = []
y_list = []

for i in range(12, 16):
    temp_masterDF = CDF.createMasterDataFrameAllEmotions(str(i), "english", "TEMPO")
    X_list.append(temp_masterDF)

    temp_initial_length = len(temp_masterDF) // 5
    temp_y = CTV.createTargetVectorALL(temp_initial_length)
    y_list.append(temp_y)

masterDF = np.concatenate(X_list)
y = np.concatenate(y_list)

print(masterDF.shape, len(y))
X_train, X_test, y_train, y_test = train_test_split(masterDF, y, stratify=y, random_state=1)

clf = svm.SVC(decision_function_shape='ovo', kernel="rbf")
clf.fit(X_train, y_train)

df = clf.predict(masterDF)
y = CTV.createTargetVectorALL(initial_length)

PS.printScores(y, df)
