import numpy as np
from sklearn import svm
import util.CreateDataFrames as CDF
import util.CreateTargetVectors as CTV
import util.printScores as PS
from sklearn.model_selection import train_test_split

K = 1

# Create features using "MEL"
masterDF_mel = CDF.createMasterDataFrameAllEmotions("11", "english", "MEL")
initial_length_mel = len(masterDF_mel) // 5
print(initial_length_mel)

for i in range(12, 16):
    K += 1
    masterDF_mel = np.concatenate((masterDF_mel, CDF.createMasterDataFrameAllEmotions(str(i), "english", "MEL")))

y_mel = CTV.createTargetVectorALL(initial_length_mel)

for i in range(12, 16):
    y_mel = np.concatenate((y_mel, CTV.createTargetVectorALL(initial_length_mel)))

# Create features using "TEMPO"
masterDF_tempo = CDF.createMasterDataFrameAllEmotions("11", "english", "TEMPO")
initial_length_tempo = len(masterDF_tempo) // 5
print(initial_length_tempo)

for i in range(12, 16):
    K += 1
    masterDF_tempo = np.concatenate((masterDF_tempo, CDF.createMasterDataFrameAllEmotions(str(i), "english", "TEMPO")))

y_tempo = CTV.createTargetVectorALL(initial_length_tempo)

for i in range(12, 16):
    y_tempo = np.concatenate((y_tempo, CTV.createTargetVectorALL(initial_length_tempo)))

# Repeat "TEMPO" feature for each corresponding frame in "MEL" features
masterDF_tempo_replicated = np.tile(masterDF_tempo, (masterDF_mel.shape[0], 1))

# Combine "MEL" and "TEMPO" features
X_mel_tempo = np.concatenate((masterDF_mel, masterDF_tempo_replicated), axis=1)
y = np.concatenate((y_mel, y_tempo))

print(X_mel_tempo.shape, len(y))
X_train, X_test, y_train, y_test = train_test_split(X_mel_tempo, y, stratify=y, random_state=1)

clf = svm.SVC(decision_function_shape='ovo', kernel="rbf")
clf.fit(X_train, y_train)

df = clf.predict(X_mel_tempo)
y = np.concatenate((y_mel, y_tempo))

PS.printScores(y, df)
