from keras.models import load_model
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import joblib

# Define data path and emotions

emotion_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

# Function to load audio data based on emotion classes
def load_audio_data(data_path):
    return librosa.load(data_path)

# Function to generate spectrograms from audio data
def generate_spectrograms(audio, num_freq_bins=128):
    audio = np.array(audio)
    spectrogram = librosa.stft(audio)
    spectrogram_db = librosa.amplitude_to_db(abs(spectrogram))

    # Ensure a consistent number of frequency bins
    if spectrogram_db.shape[1] < num_freq_bins:
        pad_width = num_freq_bins - spectrogram_db.shape[1]
        spectrogram_db = np.pad(spectrogram_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif spectrogram_db.shape[1] > num_freq_bins:
        spectrogram_db = spectrogram_db[:, :num_freq_bins]

    spectrogram_reshaped = spectrogram_db[..., np.newaxis]  # Add channel dimension
    return spectrogram_reshaped

print("Loading model...")
model = load_model("Models/CNN/CNNorg.keras")
scaler = joblib.load("scalers/scalerOrg.joblib")
data_path = "testSet/"
while True:
    select = input("Name of the audio file you wish to be predicted: ")
    data_path_temp = data_path + select
    print("Loading audio...")
    audio_data, _ = librosa.load(data_path_temp)

    print("Normalizing data...")
    spectrogram = generate_spectrograms(audio_data)

    # Normalize spectrogram
    normalized_spectrogram = scaler.fit_transform(spectrogram[:, :, 0])

    # Reshape the normalized spectrogram to add channel dimension
    normalized_spectrogram = normalized_spectrogram[..., np.newaxis]

    print("predicting...")
    y_pred = model.predict(np.expand_dims(normalized_spectrogram, axis=0))

    # Get the predicted emotion
    predicted_class_index = np.argmax(y_pred, axis=1)
    predicted_emotion = emotion_labels[predicted_class_index[0]]
    print("--------------------------------------")
    print("Predicted Emotion: ", predicted_emotion)
    print("--------------------------------------")
    empty = input("Press any button to try again\n")
"""
    predicted_class_index = np.argmax(y_pred, axis=1)
    predicted_emotion = emotion_labels[predicted_class_index[0]]
    print("Predicted Emotion: ", predicted_emotion)
"""
