import os
import librosa
import librosa.feature
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ANGRY = "Angry"
HAPPY = "Happy"
NEUTRAL = "Neutral"
SAD = "Sad"
SURPRISE = "Surprise"

emotion_categories = [ANGRY,HAPPY,NEUTRAL,SAD,SURPRISE]

if not os.path.exists("ExtractedFeatures"):
    os.mkdir("ExtractedFeatures")

#get file paths of all audios of specified category as string
def save_audio_to_array(speaker_name,category):
    audio = []
    for file in os.listdir("trainingSet/english/00"+speaker_name+ "/" + category):
        audio.append("trainingSet/english/00"+speaker_name+ "/" + category+"/"+file)

    return audio

#extract features from specified file
#then write feature values to a file
#NOT COMPLETE, DEMO VERSION
def extract_features_by_file(speaker_name, category,audio):
    #load audio, save audio file as floating point numbers to y
    y, sr = librosa.load(audio)
    # save name to write a file
    filename = audio.strip("trainingSet/english/00"+speaker_name+ "/" + category+"/").strip(".wav")
    filename = '00'+speaker_name+filename
    os.chdir("ExtractedFeatures/english/00"+speaker_name+ "/" + category)
    #####################FEATURE EXTRACTIONS#####################
    #MFCC
    a = librosa.feature.mfcc(y=y, sr=sr)
    b = []
    for i in a:
        b.append(np.average(i))
    extracted_features = {
        'mfcc':b
    }
    df = pd.DataFrame(extracted_features)
    df.to_csv(filename+'.csv')
    #to be completed
    print("CATEGORY: ",category," from: ",audio," features has been saved as: " ,filename)
    os.chdir("../../../..")

#extract features based on emotion category
#category = folder name
def extract_features_by_category(speaker_name,category):
    if not os.path.exists("ExtractedFeatures/english/00"+speaker_name+ "/" + category):
        os.mkdir("ExtractedFeatures/english/00"+speaker_name+ "/" + category)
    audios = save_audio_to_array(speaker_name, category)
    for audio in audios:
        extract_features_by_file(speaker_name,category,audio)


#extract features of an actor/actress
#speaker_name = folder name (0011,0012...)
def extract_features_by_speaker(speaker_name):
    if not os.path.exists("ExtractedFeatures/english/00" + speaker_name):
        os.mkdir("ExtractedFeatures/english/00" + speaker_name)
    for category in emotion_categories:
        extract_features_by_category(speaker_name,category)

#extract features from english speaking training set
def extract_english_features():
    if not os.path.exists("ExtractedFeatures/english/"):
        os.mkdir("ExtractedFeatures/english/")
    extract_features_by_speaker(str(11))
#    i = 10
#    while i < 20:
#        i = i + 1
#        extract_features_by_speaker(str(i))

def main():
    extract_english_features()

main()
