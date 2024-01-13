import wave
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file (replace with your file path)
#audio_file = wave.open("trainingSet/english/0011/Angry/0011_000351.wav", "rb")
#audio_file = wave.open("trainingSet/mandarin/0002/Angry/0002_000353.wav", "rb")
audio_file = wave.open("trainingSet/mandarin/0002/Surprise/0002_001403.wav", "rb")



# Get audio parameters
frame_rate = audio_file.getframerate()
frames = audio_file.getnframes()

# Read audio data as a byte string
audio_data = audio_file.readframes(frames)

# Convert audio data to a NumPy array of integers
audio_data = np.frombuffer(audio_data, dtype=np.int16)

# Create time axis in seconds
time_axis = np.linspace(0, frames / frame_rate, frames)

# Plot the waveform
plt.plot(time_axis, audio_data)

# Label the axes
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# Title the plot
plt.title("Audio Waveform")

# Show the plot
plt.show()