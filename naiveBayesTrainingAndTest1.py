import numpy as np
import sklearn
from sklearn import svm
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,confusion_matrix
import util.CreateDataFrames as CDF
import util.CreateTargetVectors as CTV
import util.printScores as PS
from sklearn.model_selection import train_test_split
import util.kBestFeatures as kBF
import util.mergeDataFrames as uDF
from sklearn.preprocessing import StandardScaler
import sklearn.feature_selection as fs
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import time
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

#masterDF = CDF.createMasterDataFrameAllEmotions("11","english","MFCC")
#masterDF = CDF.createMasterDataFrameAllEmotions("11","english","MFCC")

#alternative call to above :  CDF.createMasterDataFrame("11","english","MFCC",True,True,True,True,True)
#inital_length = len(masterDF)/5
#print(inital_length)
#for i in range(12,16):
#    K+=1
#    masterDF = np.concatenate((masterDF,CDF.createMasterDataFrameAllEmotions(str(i),"english","MFCC")))



features = ["CHROMA_VQT"]

K=20
#masterDF = uDF.getAveragedMergedDataFrames("11","english",features)
masterDF = uDF.getMergedDataFrames("11","english",features)
#alternative call to above :  CDF.createMasterDataFrame("11","english","MFCC",True,True,True,True,True)

inital_length = len(masterDF)/5
for i in range(12,18):
    K+=1
    masterDF = np.concatenate((masterDF,uDF.getMergedDataFrames(str(i),"english",features)))
    #masterDF = np.concatenate((masterDF, CDF.createMasterDataFrameAllEmotions(str(i), "english", "MFCC")))
    #masterDF = np.concatenate((masterDF,uDF.getMergedDataFramesWithKBestFeatures(str(i),"english",features,K)))
    #masterDF = np.concatenate((masterDF,uDF.getAveragedMergedDataFrames(str(i),"english",features)))

for i in range(1,8):
    K+=1
    print(i)
    #masterDF = np.concatenate((masterDF,uDF.getMergedDataFramesSpecifiedEmotions(str(i),"mandarin",features,emotions)))
    masterDF = np.concatenate(
        (masterDF, uDF.getMergedDataFrames(str(i), "mandarin", features)))



y=CTV.createTargetVectorALL(inital_length)
X =masterDF
for i in range(12,18):
    y = np.concatenate((y, CTV.createTargetVectorALL(inital_length)))

for i in range(1,8):
    y = np.concatenate((y, CTV.createTargetVectorALL(inital_length)))

#masterDF.to_csv("master",index=False)
print(masterDF.shape,len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=None,test_size=0.8)

#testDF = uDF.getAveragedMergedDataFrames("17","english",features)
testDF = uDF.getMergedDataFrames("18","english",features)
#testDF = uDF.getMergedDataFramesWithKBestFeatures("17","english",features,K)
testY = CTV.createTargetVectorALL(inital_length)
"""
#normalize data
transformer = Normalizer()
X_train = transformer.fit_transform(X_train)
X_test = transformer.transform(X_test)
testDF = transformer.transform(testDF)
"""
#normalize data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# apply same transformation to test data
X_test = scaler.transform(X_test)
testDF = scaler.transform(testDF)
"""
scaler = MinMaxScaler()
# Fit and transform the data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
testDF = scaler.transform(testDF)
"""
#select K best features
#best_features = fs.SelectKBest(fs.f_classif, k=K)
feature_selector = fs.SelectKBest(fs.f_classif, k='all')
X_train = feature_selector.fit_transform(X_train, y_train)
X_test = feature_selector.transform(X=X_test)
testDF = feature_selector.transform(X=testDF)

#traing SVM
# clf = svm.SVC(decision_function_shape='ovo', kernel= "rbf")
# clf.fit(X_train,y_train)
training_start = time.time()
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)
training_end = time.time()
print("time: ",training_end-training_start)
print("-----------TEST SPLIT--------------------")

# y_pred = clf.predict(X_test)
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

# Calculate metrics
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)
print("Overall accuracies: ")
# Print the results
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'Accuracy: {accuracy:.4f}')

# print("-------------TEST ACTOR 18-----------------")
training_start = time.time()
predictions = gnb.predict(testDF)
training_end = time.time()
print("time train", training_end-training_start)
PS.printScoresOld(testY,predictions)

# class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
# conf_matrix = confusion_matrix(testY,predictions)
# print("Confusion Matrix:")
# print("\t", *class_labels)
# for i, label in enumerate(class_labels):
#     print(label + "\t  ", conf_matrix[i])

# class_accuracies = calculate_accuracy(conf_matrix, class_labels)

# for class_label, metrics in class_accuracies.items():
#     print(f"Class: {class_label}")
#     print(f"  Accuracy: {metrics['accuracy']:.2f}")
#     print(f"  Recall: {metrics['recall']:.2f}")
#     print(f"  Precision: {metrics['precision']:.2f}")
#     f1score = 2*(metrics['recall']*metrics['precision']/(metrics['recall'] + metrics['precision']))
#     print("F1 Score: ", f1score)
#     print("-" * 20)

# precision = precision_score(testY, predictions, average='macro')
# recall = recall_score(testY, predictions, average='macro')
# f1 = f1_score(testY, predictions, average='macro')
# accuracy = accuracy_score(testY, predictions)
# print("Overall accuracies: ")
# # Print the results
# print(f'Precision: {precision:.4f}')
# print(f'Recall: {recall:.4f}')
# print(f'F1-score: {f1:.4f}')
# print(f'Accuracy: {accuracy:.4f}')
class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print("\t", *class_labels)  
for i, label in enumerate(class_labels):
    print(label + "\t  ", conf_matrix[i])

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()