import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import util.CreateDataFrames as CDF
import util.CreateTargetVectors as CTV
import util.printScores as PS
import util.mergeDataFrames as uDF
import sklearn.feature_selection as fs

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam

features = ["MFCC","CHROMA_VQT","MEL"]

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
masterDF = uDF.getMergedDataFrames("11","english",features)
#masterDF = CDF.createMasterDataFrameAllEmotions("11", "english", "MFCC")
#masterDF = uDF.getMergedDataFramesWithKBestFeatures("11","english",features,K)
#alternative call to above :  CDF.createMasterDataFrame("11","english","MFCC",True,True,True,True,True)
#inital_length = len(masterDF)/5
inital_length = 350
print(inital_length)
for i in range(12,19):
    K+=1
    masterDF = np.concatenate((masterDF,uDF.getMergedDataFrames(str(i),"english",features)))
    #masterDF = np.concatenate((masterDF, CDF.createMasterDataFrameAllEmotions(str(i), "english", "MFCC")))
    #masterDF = np.concatenate((masterDF,uDF.getMergedDataFramesWithKBestFeatures(str(i),"english",features,K)))
    #masterDF = np.concatenate((masterDF,uDF.getAveragedMergedDataFrames(str(i),"english",features)))

y=CTV.createTargetVectorALL(inital_length)
for i in range(12,19):
    y = np.concatenate((y, CTV.createTargetVectorALL(inital_length)))

#masterDF.to_csv("master",index=False)
print(masterDF.shape,len(y))

#masterDF = pad_sequences(masterDF, maxlen=200, padding='post', truncating='post',value=0)

#-----------ENCODE LABELS HERE-------------------------------#
# Convert string labels to integer labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#---------SET TRAINING-TEST SETS HERE-------------------#

X = masterDF
#X, y = make_classification(n_samples=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y,random_state=1,test_size=0.2)
#testDF = CDF.createMasterDataFrameAllEmotions("18", "english", "MFCC")
#testDF = uDF.getAveragedMergedDataFrames("18","english",features)

#---------APPLY NORMALIZATION HERE---------------------#

scaler = StandardScaler()
# fit only on training data
X_train = scaler.fit_transform(X_train)
# transformation to test data
X_test = scaler.transform(X_test)

#----------------APPLY FEATURE SELECTION HERE----------------#

feature_selector = fs.SelectKBest(fs.f_classif, k='all')
X_train = feature_selector.fit_transform(X_train, y_train)
X_test = feature_selector.transform(X=X_test)

#-------------------TRAIN THE MODEL HERE---------------------#

# Build the MLP model
model = Sequential()

# Input layer
model.add(Dense(units=256, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))
# Hidden layers
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))

# Output layer for multi-class classification
model.add(Dense(units=5, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=len(X_train), validation_split=0.1)

#----------------------PRINT THE RESULTS HERE---------------------------#

print("-----------TEST SPLIT--------------------")

y_pred = model.predict(X_test)

#decode labels
predicted_labels_indices = np.argmax(y_pred, axis=1)
y_pred = label_encoder.inverse_transform(predicted_labels_indices)

y_test = label_encoder.inverse_transform(y_test)

PS.printScores(y_test,y_pred)

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




