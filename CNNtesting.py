from keras.models import load_model
# Drive'a bağlanın
#drive.mount("/content/drive")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import librosa
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score,accuracy_score
import joblib

# Define data path and emotions
data_path = "/content/drive/MyDrive/AraProje/Workspace/trainingSet/english/0020"
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

def generate_mel_spectrograms(audio_data, num_freq_bins=128):
    mel_spectrograms = []
    for audio in audio_data:
        # Generate mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, n_mels=num_freq_bins)
        mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram)

        # Ensure a consistent number of frequency bins
        if mel_spectrogram_db.shape[1] < num_freq_bins:
            pad_width = num_freq_bins - mel_spectrogram_db.shape[1]
            mel_spectrogram_db = np.pad(mel_spectrogram_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        elif mel_spectrogram_db.shape[1] > num_freq_bins:
            mel_spectrogram_db = mel_spectrogram_db[:, :num_freq_bins]

        # Reshape for compatibility with the model
        mel_spectrogram_reshaped = mel_spectrogram_db.reshape(1, mel_spectrogram_db.shape[0], mel_spectrogram_db.shape[1], 1)
        mel_spectrograms.append(mel_spectrogram_reshaped)

    return np.concatenate(mel_spectrograms, axis=0)

def generate_dual_spectrograms(audio_data, num_freq_bins=512, max_frames=None):
    spectrograms = []

    for audio in audio_data:
        # Generate the first spectrogram (MFCC)
        spectrogram_1 = librosa.feature.mfcc(y=audio, n_mfcc= num_freq_bins)
        spectrogram_db_1 = librosa.amplitude_to_db(spectrogram_1)

        # Generate the second spectrogram (Mel spectrogram)
        spectrogram_2 = librosa.feature.melspectrogram(y=audio, n_mels=num_freq_bins)
        spectrogram_db_2 = librosa.amplitude_to_db(abs(spectrogram_2))

        # Pad or truncate frames to a fixed length
        if max_frames:
            spectrogram_db_1 = pad_sequences([spectrogram_db_1.T], maxlen=max_frames, padding='post', truncating='post').T
            spectrogram_db_2 = pad_sequences([spectrogram_db_2.T], maxlen=max_frames, padding='post', truncating='post').T

        # Ensure a consistent number of frequency bins for both spectrograms
        max_freq_bins = max(spectrogram_db_1.shape[1], spectrogram_db_2.shape[1])

        if spectrogram_db_1.shape[1] < max_freq_bins:
            pad_width_1 = max_freq_bins - spectrogram_db_1.shape[1]
            spectrogram_db_1 = np.pad(spectrogram_db_1, pad_width=((0, 0), (0, pad_width_1)), mode='constant')
        elif spectrogram_db_1.shape[1] > max_freq_bins:
            spectrogram_db_1 = spectrogram_db_1[:, :max_freq_bins]

        if spectrogram_db_2.shape[1] < max_freq_bins:
            pad_width_2 = max_freq_bins - spectrogram_db_2.shape[1]
            spectrogram_db_2 = np.pad(spectrogram_db_2, pad_width=((0, 0), (0, pad_width_2)), mode='constant')
        elif spectrogram_db_2.shape[1] > max_freq_bins:
            spectrogram_db_2 = spectrogram_db_2[:, :max_freq_bins]
        # Stack both spectrograms along the channel axis
        dual_spectrogram = np.stack([spectrogram_db_1, spectrogram_db_2], axis=-1)

        # Reshape for compatibility with the model
        dual_spectrogram_reshaped = dual_spectrogram.reshape(1, dual_spectrogram.shape[0], dual_spectrogram.shape[1],2)
        spectrograms.append(dual_spectrogram_reshaped)

    return np.concatenate(spectrograms, axis=0)

print("loading model...")
model = load_model("/content/drive/MyDrive/AraProje/Workspace/CNNsmall.keras")

print("loading audio...")
audio_data , emotions = load_audio_data(data_path, emo)

print("normalizing data...")
spectrograms = generate_spectrograms(audio_data)
# Flatten the spectrograms before normalization
flattened_spectrograms = spectrograms.reshape(spectrograms.shape[0], -1)

scaler = joblib.load('/content/drive/MyDrive/AraProje/Workspace/scaler.joblib')
# Normalize flattened spectrograms
normalized_spectrograms = scaler.fit_transform(flattened_spectrograms)

# Reshape the normalized spectrograms back to their original shape
normalized_spectrograms = normalized_spectrograms.reshape(spectrograms.shape)


y_pred = model.predict(normalized_spectrograms)

class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
# Create confusion matrix
y_pred_temp = y_pred

numerical_labels = np.argmax(y_pred, axis=1)

# Define your mapping between numerical and string labels
numerical_to_string_mapping = {0: 'Angry', 1: 'Happy', 2: 'Neutral',3:'Sad',4:'Surprise'}  # Update with your actual mapping

# Convert numerical labels to string labels
string_labels = [numerical_to_string_mapping[label] for label in numerical_labels]

print("Predicted String Labels:", string_labels)


cm = confusion_matrix(emotions, string_labels)

# Plot confusion matrix using Seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels,
            yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
class_accuracies = calculate_accuracy(cm, class_labels)

for class_label, metrics in class_accuracies.items():
    print(f"Class: {class_label}")
    print(f"  Accuracy: {metrics['accuracy']:.2f}")
    print(f"  Recall: {metrics['recall']:.2f}")
    print(f"  Precision: {metrics['precision']:.2f}")
    f1score = 2*(metrics['recall']*metrics['precision']/(metrics['recall'] + metrics['precision']))
    print("F1 Score: ", f1score)
    print("-" * 20)
