import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.feature_selection as skl
import pandas as pd
from sklearn import svm

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


masterDF = createMasterDataFrameMFCC("12")
for i in range(13,16):
    masterDF = np.concatenate((masterDF,createMasterDataFrameMFCC(str(i))))
for i in range(17,19):
    masterDF = np.concatenate((masterDF, createMasterDataFrameMFCC(str(i))))
for i in range(20,21):
    masterDF = np.concatenate((masterDF, createMasterDataFrameMFCC(str(i))))
y=createTargetVector()
for i in range(12,18):
    y = np.concatenate((y, createTargetVector()))
#masterDF.to_csv("master",index=False)
print(masterDF.shape,len(y))

lin_clf = svm.LinearSVC(dual="auto")
lin_clf.fit(masterDF, y)
dec = lin_clf.decision_function(masterDF)

masterDF = createMasterDataFrameMFCC("15")
print(masterDF.shape)
df = lin_clf.predict(masterDF)

y = createTargetVector()
scores=[0,0,0,0,0]
for j in range(0,1745):
    if y[j]==df[j] and y[j] == "Angry":
        scores[0] = scores[0]+1
    elif y[j]==df[j] and y[j] == "Happy":
        scores[1] = scores[1]+1
    elif y[j]==df[j] and y[j] == "Neutral":
        scores[2] = scores[2]+1
    elif y[j]==df[j] and y[j] == "Sad":
        scores[3] = scores[3]+1
    elif y[j]==df[j] and y[j] == "Surprise":
        scores[4] = scores[4]+1


print("angry score:",scores[0]/350*100)

print("happy score:", scores[1]/350*100)

print("neutral score:", scores[2]/350*100)

print("sad score:", scores[3]/350*100)

print("surprise score:", scores[4]/350*100)