import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import util.CreateDataFrames as CDF
import util.CreateTargetVectors as CTV
import util.printScores as PS
import util.mergeDataFrames as uDF
import sklearn.feature_selection as fs
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from keras.callbacks import *


features = ["MFCC","MEL","RMS","CHROMA_STFT","TEMPO"]

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
emotions = [True,True,True,False,False]
#masterDF = uDF.getAveragedMergedDataFrames("11","english",features)
masterDF = uDF.getMergedDataFrames("11","english",features)
#masterDF = uDF.getMergedDataFramesSpecifiedEmotions("11","english",features,emotions)
#masterDF = CDF.createMasterDataFrameAllEmotions("11", "english", "MFCC")
#masterDF = uDF.getMergedDataFramesWithKBestFeatures("11","english",features,K)
#alternative call to above :  CDF.createMasterDataFrame("11","english","MFCC",True,True,True,True,True)
#inital_length = len(masterDF)/5
inital_length = 350
print(inital_length)
for i in range(12,18):
    K+=1
    masterDF = np.concatenate((masterDF,uDF.getMergedDataFrames(str(i),"english",features)))
    #masterDF = np.concatenate((masterDF, CDF.createMasterDataFrameAllEmotions(str(i), "english", "MFCC")))
    #masterDF = np.concatenate((masterDF,uDF.getMergedDataFramesWithKBestFeatures(str(i),"english",features,K)))
    #masterDF = np.concatenate((masterDF,uDF.getAveragedMergedDataFrames(str(i),"english",features)))
"""""
for i in range(1,10):
    K+=1
    print(i)
    #masterDF = np.concatenate((masterDF,uDF.getMergedDataFramesSpecifiedEmotions(str(i),"mandarin",features,emotions)))
    masterDF = np.concatenate(
        (masterDF, uDF.getMergedDataFrames(str(i), "mandarin", features)))
"""

"""
y=CTV.createTargetVector(inital_length,emotions[0], emotions[1], emotions[2],emotions[3],emotions[4])
for i in range(12,18):
    y = np.concatenate((y, CTV.createTargetVector(inital_length,emotions[0], emotions[1], emotions[2],emotions[3],emotions[4])))
for i in range(1,10):
    y = np.concatenate((y, CTV.createTargetVector(inital_length,emotions[0], emotions[1], emotions[2],emotions[3],emotions[4])))

"""
y=CTV.createTargetVectorALL(inital_length)
for i in range(12,18):
    y = np.concatenate((y, CTV.createTargetVectorALL(inital_length)))
"""
for i in range(1,10):
    y = np.concatenate((y, CTV.createTargetVectorALL(inital_length)))
"""
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
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded,random_state=1,test_size=0.3,shuffle=True)

testDF = uDF.getMergedDataFrames("18","english",features)
#testY = CTV.createTargetVector(inital_length,emotions[0], emotions[1], emotions[2],emotions[3],emotions[4])

testY = CTV.createTargetVectorALL(inital_length)
"""""
testDF = uDF.getMergedDataFrames("18", "english", features)
#testDF = uDF.getAveragedMergedDataFrames("18","english",features)
"""
#---------APPLY NORMALIZATION HERE---------------------#

#scaler = StandardScaler()
# fit only on training data
#X_train = scaler.fit_transform(X_train)
# transformation to test data
#X_test = scaler.transform(X_test)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()
# Fit and transform the data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
testDF = scaler.transform(testDF)

#----------------APPLY FEATURE SELECTION HERE----------------#

feature_selector = fs.SelectKBest(fs.f_classif, k='all')
X_train = feature_selector.fit_transform(X_train, y_train)
X_test = feature_selector.transform(X=X_test)
testDF = feature_selector.transform(X=testDF)

#-------------------TRAIN THE MODEL HERE---------------------#

# Build the MLP model
model = Sequential()
callback = EarlyStopping(monitor='loss',patience=3)
# Input layer
model.add(Dense(units=128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.05))
# Hidden layers
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.1))

# Output layer for multi-class classification
model.add(Dense(units=5, activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.3)

#----------------------PRINT THE RESULTS HERE---------------------------#

print("-----------TEST SPLIT--------------------")

#y_pred = model.predict(X_test)

y_pred = model.predict(testDF)

#decode labels
predicted_labels_indices = np.argmax(y_pred, axis=1)
y_pred = label_encoder.inverse_transform(predicted_labels_indices)

#y_test = label_encoder.inverse_transform(y_test)

PS.printScores(testY,y_pred)

class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
conf_matrix = confusion_matrix(testY,y_pred)
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

#----------PLOT METRICS CHANGE OVER ITERATIONS---------------

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#-----------PLOT CONFUSION MATRIX-------------------
# Create confusion matrix
cm = confusion_matrix(testY, y_pred)

# Plot confusion matrix using Seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["ANGRY","HAPPY","NEUTRAL","SAD","SURPRISE"],
            yticklabels=["ANGRY","HAPPY","NEUTRAL","SAD","SURPRISE"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

"""
print("-----------TEST ACTOR 18--------------------")

y_pred_18 = model.predict(testDF)

#decode labels
predicted_labels_indices = np.argmax(y_pred_18, axis=1)
y_pred_18 = label_encoder.inverse_transform(predicted_labels_indices)

PS.printScores(testY,y_pred_18)

class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
conf_matrix = confusion_matrix(testY,y_pred_18)
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
"""
#-------------SAVE MODEL IF REQUESTED-----------------#

decision = bool(input("Would you like to save the model: "))
if not os.path.exists("Models"):
    os.mkdir("Models")
if not os.path.exists("Models/MLP"):
    os.mkdir("Models/MLP")

if decision:
    name= input("Model name: ")
    model.save('Models/MLP/sucess'+name+'.keras')  # The file needs to end with the .keras extension