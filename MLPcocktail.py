import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.feature_selection as skl
import pandas as pd
import sklearn.metrics
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import util.CreateDataFrames as CDF
import util.CreateTargetVectors as CTV
import util.printScores as PS
import util.mergeDataFrames as uDF
import sklearn.feature_selection as fs

features = ["MFCC"]

def calculate_accuracy(conf_matrix, class_labels):

    results = {}

    for i, class_label in enumerate(class_labels):
        # True positives, false positives, true negatives, and false negatives
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[i, :]) - tp
        tn = np.sum(np.diag(conf_matrix)) - (tp + np.sum(conf_matrix[:, i]))
        fn = np.sum(conf_matrix[:, i]) - tp

        # Calculate accuracy, recall, and precision
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0

        results[class_label] = {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision
        }

    return results

K=15
#masterDF = uDF.getAveragedMergedDataFrames("11","english",features)
#masterDF = uDF.getMergedDataFrames("11","english",features)
masterDF = CDF.createMasterDataFrameAllEmotions("11","english","MFCC")
#masterDF = uDF.getMergedDataFramesWithKBestFeatures("11","english",features,K)
#alternative call to above :  CDF.createMasterDataFrame("11","english","MFCC",True,True,True,True,True)
inital_length = len(masterDF)/5
print(inital_length)
for i in range(12,18):
    K+=1
    #masterDF = np.concatenate((masterDF,uDF.getMergedDataFrames(str(i),"english",features)))

    #masterDF = np.concatenate((masterDF,uDF.getMergedDataFramesWithKBestFeatures(str(i),"english",features,K)))
    #masterDF = np.concatenate((masterDF,uDF.getAveragedMergedDataFrames(str(i),"english",features)))

y=CTV.createTargetVectorALL(inital_length)
for i in range(12,18):
    y = np.concatenate((y, CTV.createTargetVectorALL(inital_length)))

#masterDF.to_csv("master",index=False)
print(masterDF.shape,len(y))

X = masterDF
#X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1,test_size=0.2)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Don't cheat - fit only on training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# apply same transformation to test data
X_test = scaler.transform(X_test)
model = MLPClassifier(alpha = 0.0001, batch_size = 256, epsilon = 1e-08, hidden_layer_sizes = (16,), learning_rate = 'adaptive', max_iter =500)

#testDF = uDF.getAveragedMergedDataFrames("17","english",features)
#testDF = uDF.getMergedDataFrames("18","english",features)
testDF = CDF.createMasterDataFrameAllEmotions("18","english","MFCC")
#testDF = uDF.getMergedDataFramesWithKBestFeatures("17","english",features,K)
testY = CTV.createTargetVectorALL(inital_length)
testDF = scaler.transform(testDF)

feature_selector = fs.SelectKBest(fs.f_classif, k='all')
X_train = feature_selector.fit_transform(X_train, y_train)
X_test = feature_selector.transform(X=X_test)
testDF = feature_selector.transform(X=testDF)

model.fit(X_train, y_train)

print("-----------TEST SPLIT--------------------")

y_pred = model.predict(X_test)
PS.printScoresOld(y_test,y_pred)

class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
conf_matrix = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print("\t", *class_labels)
for i, label in enumerate(class_labels):
    print(label + "\t  ", conf_matrix[i])

class_accuracies = calculate_accuracy(conf_matrix, class_labels)

for class_label, metrics in class_accuracies.items():
    print(f"Class: {class_label}")
    print(f"  Accuracy: {metrics['accuracy']:.2f}")
    print(f"  Recall: {metrics['recall']:.2f}")
    print(f"  Precision: {metrics['precision']:.2f}")
    f1score = 2*(metrics['recall']*metrics['precision']/(metrics['recall'] + metrics['precision']))
    print("F1 Score: ", f1score)
    print("-" * 20)


print("-------------TEST ACTOR 18-----------------")
predictions = model.predict(testDF)

PS.printScoresOld(testY,predictions)

class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
conf_matrix = confusion_matrix(testY,predictions)
print("Confusion Matrix:")
print("\t", *class_labels)
for i, label in enumerate(class_labels):
    print(label + "\t  ", conf_matrix[i])

class_accuracies = calculate_accuracy(conf_matrix, class_labels)

for class_label, metrics in class_accuracies.items():
    print(f"Class: {class_label}")
    print(f"  Accuracy: {metrics['accuracy']:.2f}")
    print(f"  Recall: {metrics['recall']:.2f}")
    print(f"  Precision: {metrics['precision']:.2f}")
    f1score = 2*(metrics['recall']*metrics['precision']/(metrics['recall'] + metrics['precision']))
    print("F1 Score: ", f1score)
    print("-" * 20)

