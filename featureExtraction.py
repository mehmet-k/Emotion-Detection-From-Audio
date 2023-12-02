import os

import librosa
import librosa.feature
import numpy
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists("ExtractedFeatures/english/"):
    os.mkdir("ExtractedFeatures/english/")

ANGRY = "Angry"
HAPPY = "Happy"
NEUTRAL = "Neutral"
SAD = "Sad"
SURPRISE = "Surprise"

emotion_categories = [ANGRY,HAPPY,NEUTRAL,SAD,SURPRISE]

def save_audio_to_array(filename,category):
    audio = []
    for file in os.listdir('trainingSet/english/00'+filename+"/"+category):
        audio.append("trainingSet/english/00"+filename+"/"+category+"/"+file)

    return audio

def extract_feautes_by_file(speaker_name, category,path):
    filename = str(path).strip("trainingSet/english/00" + speaker_name + "/" + category + "/")
    filename = filename.strip(".wav")
    #save name to write a file
    #save audio file as floating point numbers to y
    y, sr = librosa.load(path)
    #####################FEATUR EXTRACTIONS#####################
    #MFCC
    a = librosa.feature.mfcc(y=y, sr=sr)
    os.chdir("ExtractedFeatures/english/00" + speaker_name + "/" + category + "/")
    #save extracted mfcc features to to file
    numpy.savetxt(filename,a)
    os.chdir("../../../..")

def extract_features_by_category(speaker_name,category):
    if not os.path.exists("ExtractedFeatures/english/00"+speaker_name+ "/" + category):
        os.mkdir("ExtractedFeatures/english/00"+speaker_name+ "/" + category)
    paths = np.array(save_audio_to_array(speaker_name, category))
    for i in range(len(paths)):
        extract_feautes_by_file(speaker_name,category,paths[i])

def extract_features_by_speaker(speaker_name):
    if not os.path.exists("ExtractedFeatures/english/00" + speaker_name):
        os.mkdir("ExtractedFeatures/english/00" + speaker_name)
    for i in range(len(emotion_categories)):
        extract_features_by_category(speaker_name,emotion_categories[i])

def main():
    extract_features_by_speaker(str(11))
    #i=10
    #while i<20:
    #    i=i+1
    #    extract_features_by_speaker(str(i))
    #
main()
