import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.feature_selection as skl
import pandas as pd
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report



import util.CreateDataFrames as CDF
import util.CreateTargetVectors as CTV
import util.printScores as PS

def calculate_accuracy(conf_matrix, class_labels):
#     accuracies = []

#     for i in range(len(class_labels)):
#         correct_predictions = conf_matrix[i, i]
#         total_predictions = np.sum(conf_matrix[i, :])
#         accuracy = correct_predictions / total_predictions if total_predictions != 0 else 0
#         accuracies.append(accuracy)

#     return accuracies
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
K=1
#masterDF = CDF.createMasterDataFrameAllEmotions("11","english","MFCC")
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
model = MLPClassifier(alpha = 0.01, batch_size = 256, epsilon = 1e-08, hidden_layer_sizes = (300), learning_rate = 'adaptive', max_iter = 500)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#y-test y-pred
y=CTV.createTargetVectorALL(inital_length)

#PS.printScores(y,y_pred)
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

emotions = ['angry', 'happy', 'neutral', 'sad', 'surprise']
scores = [scores[0]/count[0]*100, scores[1]/count[1]*100, scores[2]/count[2]*100, scores[3]/count[3]*100, scores[4]/count[4]*100]

plt.bar(emotions, scores, color=['#FFB6C1', '#FFD700', '#98FB98', '#ADD8E6', '#FFA07A'])
plt.xlabel('Emotions', fontweight='bold')
plt.ylabel('Scores', fontweight='bold')
plt.title('Emotion Scores')
plt.ylim(0, 100)  

plt.show()    
class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
conf_matrix = confusion_matrix(y_test, y_pred)
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
    print("-" * 20)  # Add a separator between classes


