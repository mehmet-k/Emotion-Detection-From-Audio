from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import make_classification
import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import util.CreateTargetVectors as CTV
import util.printScores as PS
from sklearn.model_selection import train_test_split
import util.kBestFeatures as kBF
import util.mergeDataFrames as uDF
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler



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
features = ["MFCC","RMS"]

masterDF = uDF.getMergedDataFrames("11","english",features)
#alternative call to above :  CDF.createMasterDataFrame("11","english","MFCC",True,True,True,True,True)

inital_length = len(masterDF)/5
for i in range(12,18):
   
    masterDF = np.concatenate((masterDF,uDF.getMergedDataFrames(str(i),"english",features)))
    #masterDF = np.concatenate((masterDF, CDF.createMasterDataFrameAllEmotions(str(i), "english", "MFCC")))
    #masterDF = np.concatenate((masterDF,uDF.getMergedDataFramesWithKBestFeatures(str(i),"english",features,K)))
    #masterDF = np.concatenate((masterDF,uDF.getAveragedMergedDataFrames(str(i),"english",features)))

for i in range(1,8):
   
    print(i)
    masterDF = np.concatenate(
        (masterDF, uDF.getMergedDataFrames(str(i), "mandarin", features)))



y=CTV.createTargetVectorALL(inital_length)
X =masterDF
for i in range(12,18):
    y = np.concatenate((y, CTV.createTargetVectorALL(inital_length)))

for i in range(1,8):
    y = np.concatenate((y, CTV.createTargetVectorALL(inital_length)))



for i in range(len(y)):
    if y[i] == 'Surprise' or y[i] == 'Happy':
        y[i] = 'Positive'
    elif y[i] == 'Angry' or y[i] == 'Sad':
        y[i] = 'Negative'

X = masterDF
#X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# apply same transformation to test data
X_test = scaler.transform(X_test)




gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

class_labels = ["Positive","Neutral","Negative"]
print(y_test)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


"""




precision = precision_score(y_test,y_pred , average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)
print("Overall accuracies: ")
# Print the results
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'Accuracy: {accuracy:.4f}')

"""