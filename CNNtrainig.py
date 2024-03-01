import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import librosa
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import joblib
from keras.preprocessing.sequence import pad_sequences
import librosa

# Define data path and emotions
data_path = "/content/drive/MyDrive/AraProje/Workspace/trainingSet/english/all/"
emo = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

# Function to load audio data based on emotion classes
def load_audio_data(data_path, emotions):
    audio_data = []
    emotion_labels = []
    for emotion in emotions:
        emotion_path = os.path.join(data_path, emotion)
        for filename in os.listdir(emotion_path):
            if filename.endswith(".wav"):
                audio_file = os.path.join(emotion_path, filename)
                audio, _ = librosa.load(audio_file)
                audio_data.append(audio)
                emotion_labels.append(emotion)
    return audio_data, emotion_labels

# Function to generate spectrograms from audio data
def generate_spectrograms(audio_data, num_freq_bins=128):
    spectrograms = []
    for audio in audio_data:
        spectrogram = librosa.stft(audio)
        spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))
        # Ensure a consistent number of frequency bins
        if spectrogram_db.shape[1] < num_freq_bins:
            pad_width = num_freq_bins - spectrogram_db.shape[1]
            spectrogram_db = np.pad(spectrogram_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        elif spectrogram_db.shape[1] > num_freq_bins:
            spectrogram_db = spectrogram_db[:, :num_freq_bins]
        spectrogram_reshaped = spectrogram_db.reshape(1, spectrogram_db.shape[0], spectrogram_db.shape[1], 1)
        spectrograms.append(spectrogram_reshaped)
    return np.concatenate(spectrograms, axis=0)






# Load audio data and generate spectrograms
audio_data = []
emotions = []

audio_data , emotions = load_audio_data(data_path, emo)
"""
for i in range(12,18):
  print("loading audio data: ", i)
  data_path_temp = data_path
  data_path_temp = data_path_temp + str(i)
  audio_data_temp , emotions_temp = load_audio_data(data_path_temp, emo)
  audio_data.append(audio_data_temp)
  emotions.append(emotions_temp)
  print("loaded audio data: ",i)

emotions = np.concatenate(emotions).ravel()
audio_data = np.concatenate(audio_data).ravel()
"""
print("creating spectograms...")
#spectrograms = generate_dual_spectrograms(audio_data, num_freq_bins=128, max_frames=128)
spectrograms = generate_spectrograms(audio_data)
# Flatten the spectrograms before normalization
flattened_spectrograms = spectrograms.reshape(spectrograms.shape[0], -1)

print("normalizing data...")
# Normalize flattened spectrograms
scaler = StandardScaler()
normalized_spectrograms = scaler.fit_transform(flattened_spectrograms)

print("reshaping...")
# Reshape the normalized spectrograms back to their original shape
normalized_spectrograms = normalized_spectrograms.reshape(spectrograms.shape)

# ...

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(normalized_spectrograms, emotions, stratify=emotions, test_size=0.2, random_state=None,shuffle = True)

# # Normalize spectrograms
# scaler = StandardScaler()
# spectrograms = scaler.fit_transform(spectrograms)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(spectrograms, emotions, stratify=emotions, test_size=0.2, random_state=42)

# Define the CNN model architecture
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(spectrograms.shape[1], spectrograms.shape[2], 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))  # 5 classes for emotions

print("compiling model...")
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
# Encode emotion labels
label_encoder = LabelEncoder()
encoded_y_train = label_encoder.fit_transform(y_train)
encoded_y_test = label_encoder.transform(y_test)

# Convert encoded labels to one-hot encoding
one_hot_y_train = to_categorical(encoded_y_train)
one_hot_y_test = to_categorical(encoded_y_test)

# ...

# Train the model
model.fit(X_train, one_hot_y_train, epochs=4, validation_data=(X_test, one_hot_y_test),batch_size=1)
# Train the model
# model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
# test_loss, test_acc = model.evaluate(X_test, y_test)
test_loss, test_acc = model.evaluate(X_test, one_hot_y_test)

print('Test accuracy:', test_acc)

model.save('/content/drive/MyDrive/AraProje/Workspace/original.keras')  # The file needs to end with the .keras extension
joblib.dump(scaler, '/content/drive/MyDrive/AraProje/Workspace/scalerOriginal.joblib')
