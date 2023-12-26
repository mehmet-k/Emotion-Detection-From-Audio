from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import os
import numpy as np
import pandas as pd

def createTargetVector():
    vector = []
    for i in range(0,349):
        vector.append('Angry')
    for i in range(0,349):
        vector.append('Happy')
    for i in range(0,349):
        vector.append('Neutral')
    for i in range(0,349):
        vector.append('Sad')
    for i in range(0,349):
        vector.append('Surprise')
    return vector

def createMasterDataFrameMFCC(speaker_name):
    os.chdir("ExtractedFeatures/english/00" + speaker_name)
    # os.chdir("K_best_features/after.csv")
    data_angry = pd.read_csv('AngryMFCC.csv')
    data_angry = data_angry.to_numpy()

    data_happy = pd.read_csv('HappyMFCC.csv')
    data_happy = data_happy.to_numpy()

    data_neutral = pd.read_csv('NeutralMFCC.csv')
    data_neutral = data_neutral.to_numpy()

    data_sad = pd.read_csv('SadMFCC.csv')
    data_sad = data_sad.to_numpy()

    data_surprise = pd.read_csv('SurpriseMFCC.csv')
    data_surprise = data_surprise.to_numpy()

    master_data_frame = data_angry
    master_data_frame = np.concatenate((master_data_frame, data_happy))
    master_data_frame = np.concatenate((master_data_frame, data_neutral))
    master_data_frame = np.concatenate((master_data_frame, data_sad))
    master_data_frame = np.concatenate((master_data_frame, data_surprise))

    # master_data_frame  = pd.DataFrame(master_data_frame)
    os.chdir("../../..")
    return master_data_frame

def createMasterDataFrame(speaker_name):
    os.chdir("ExtractedFeatures/english/00"+speaker_name)
    #os.chdir("K_best_features/after.csv")
    data_angry = pd.read_csv('Angry.csv')
    data_angry = data_angry.to_numpy()

    data_happy = pd.read_csv('Happy.csv')
    data_happy = data_happy.to_numpy()

    data_neutral = pd.read_csv('Neutral.csv')
    data_neutral = data_neutral.to_numpy()

    data_sad = pd.read_csv('Sad.csv')
    data_sad = data_sad.to_numpy()

    data_surprise = pd.read_csv('Surprise.csv')
    data_surprise = data_surprise.to_numpy()

    master_data_frame = data_angry
    master_data_frame = np.concatenate((master_data_frame,data_happy))
    master_data_frame = np.concatenate((master_data_frame,data_neutral))
    master_data_frame = np.concatenate((master_data_frame, data_sad))
    master_data_frame = np.concatenate((master_data_frame, data_surprise))

    #master_data_frame  = pd.DataFrame(master_data_frame)
    os.chdir("../../..")
    return master_data_frame

y = createTargetVector()
masterDF = createMasterDataFrameMFCC("11")
for i in range(12,16):
    masterDF = np.concatenate((masterDF,createMasterDataFrameMFCC(str(i))))

for i in range(12,16):
    y = np.concatenate((y, createTargetVector()))

X = masterDF
#X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)
model = MLPClassifier(alpha = 0.01, batch_size = 256, epsilon = 1e-08, hidden_layer_sizes = (300,), learning_rate = 'adaptive', max_iter = 500)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
scores=[0,0,0,0,0]
print(len(y_pred),y_pred)
for j in range(0,len(y_pred)):
    if y_test[j]==y_pred[j] and y_test[j] == "Angry":
        scores[0] = scores[0]+1
    elif y_test[j]==y_pred[j] and y_test[j] == "Happy":
        scores[1] = scores[1]+1
    elif y_test[j]==y_pred[j] and y_test[j] == "Neutral":
        scores[2] = scores[2]+1
    elif y_test[j]==y_pred[j] and y_test[j] == "Sad":
        scores[3] = scores[3]+1
    elif y_test[j]==y_pred[j] and y_test[j] == "Surprise":
        scores[4] = scores[4]+1
count = [0,0,0,0,0]
for j in range(0,len(y_pred)):
    if y_test[j]== "Angry":
        count[0] += 1
    elif y_test[j]== "Happy":
        count[1] += 1
    elif y_test[j]== "Neutral":
        count[2] += 1
    elif y_test[j]== "Sad":
        count[3] += 1
    elif y_test[j]== "Surprise":
        count[4] += 1
for i in range(0,len(count)):
    print(count[i])

print("angry score:",scores[0]/count[0]*100)

print("happy score:", scores[1]/count[1]*100)

print("neutral score:", scores[2]/count[2]*100)

print("sad score:", scores[3]/count[3]*100)

print("surprise score:", scores[4]/count[4]*100)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))
