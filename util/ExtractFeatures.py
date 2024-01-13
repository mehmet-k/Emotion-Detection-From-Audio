import os
import librosa
import librosa.feature
import numpy as np
import pandas as pd

"""
    to create a custom feature extraction function, just replace the code between these lines:
    #####################FEATURE EXTRACTIONS#####################
    # MEL
    a = librosa.feature.mfcc(y=y, sr=sr)
    a = saveFeaturesToDictionary(a)
    #feature extraction end
    
    for example change librosa.feature.mfcc with, lbirosa.feature.chroma_stft
    for compatiblity naming of the function should be as:
    extractOnly+MOD
"""

"""
    reminder of directory structure for feature use:
    audio files should be stored as: trainingSet/<<language>>/<<speaker_name>>/<<emotion_category>>
"""

ANGRY = "Angry"
HAPPY = "Happy"
NEUTRAL = "Neutral"
SAD = "Sad"
SURPRISE = "Surprise"

#to do simpler iteration, no other practical use
emotion_categories = [ANGRY,HAPPY,NEUTRAL,SAD,SURPRISE]

"""
    this takes an average of every vector in matrix and returns an averaged array
"""
def saveFeaturesToDictionary(feature_array):
    b = []
    for i in feature_array:
        b.append(np.average(i))
    x = [b]
    return x

"""
    language: english or mandarin
    speaker_name: name of actor or actress (format: 0001, 0002...... 0020)
    category: emotion category
    audio: audio file to be extracted
"""

def extractOnlyMEL(language,speaker_name, category,audio):
    y, sr = librosa.load(audio)
    # save name to write a file
    os.chdir("ExtractedFeatures/" + language + "/00" + speaker_name + "/")
    #####################FEATURE EXTRACTIONS#####################
    # MEL
    a = librosa.feature.mfcc(y=y, sr=sr)
    a = saveFeaturesToDictionary(a)
    #feature extraction end
    df = pd.DataFrame(a)
    df.to_csv(category + 'MEL.csv', mode='a', header=False, index=False)
    # to be completed
    print("CATEGORY: ", category, " from: ", audio, " features has been saved with mod: MEL")
    os.chdir("../../..")

def extractOnlyMFCC(language,speaker_name, category,audio):
    y, sr = librosa.load(audio)
    # save name to write a file
    os.chdir("ExtractedFeatures/" + language + "/00" + speaker_name + "/")
    #####################FEATURE EXTRACTIONS#####################
    # MFCC
    a = librosa.feature.mfcc(y=y, sr=sr)
    a = saveFeaturesToDictionary( a)
    #feature extraction end
    df = pd.DataFrame(a)
    df.to_csv(category + 'MFCC.csv', mode='a', header=False, index=False)
    # to be completed
    print("CATEGORY: ", category, " from: ", audio, " features has been saved with mod MFCC")
    os.chdir("../../..")

def extractOnlyTEMPO(language,speaker_name, category,audio):
    y, sr = librosa.load(audio)
    # save name to write a file
    os.chdir("ExtractedFeatures/" + language + "/00" + speaker_name + "/")
    #####################FEATURE EXTRACTIONS#####################
    # MFCC
    a = librosa.feature.tempo(y=y, sr=sr)
    a = saveFeaturesToDictionary( a)
    #feature extraction end
    df = pd.DataFrame(a)
    df.to_csv(category + 'TEMPO.csv', mode='a', header=False, index=False)
    # to be completed
    print("CATEGORY: ", category, " from: ", audio, " features has been saved with mod TEMPO")
    os.chdir("../../..")


def save_audio_to_array(language,speaker_name,category):
    audio = []
    for file in os.listdir("trainingSet/"+language+"/00"+speaker_name+ "/" + category):
        audio.append("trainingSet/"+language+"/00"+speaker_name+ "/" + category+"/"+file)

    return audio

def extract_features_by_category_with_mod(language,speaker_name,category,mod):
    audios = save_audio_to_array(language,speaker_name, category)
    df = pd.DataFrame()
    if mod == "MFCC":
        df.to_csv(category + 'MFCC.csv', mode='a', header=False, index=False)
        for audio in audios:
            extractOnlyMFCC(language,speaker_name,category,audio)
    elif mod == "MEL":
        df.to_csv(category + 'MEL.csv', mode='a', header=False, index=False)
        for audio in audios:
            extractOnlyMEL(language,speaker_name,category,audio)
    elif mod == "TEMPO":
        df.to_csv(category + 'TEMPO.csv', mode='a', header=False, index=False)
        for audio in audios:
            extractOnlyTEMPO(language,speaker_name,category,audio)
        
    #if new functions added for feature extraction, continue this if block
    #with the same structure

#extract features of an actor/actress
#speaker_name = folder name (0011,0012...)
def extract_features_by_speaker_with_mod(language,speaker_name,mod):
    if not os.path.exists("ExtractedFeatures/"+language+"/00" + speaker_name):
        os.mkdir("ExtractedFeatures/"+language+"/00" + speaker_name)
    for category in emotion_categories:
        extract_features_by_category_with_mod(language,speaker_name,category,mod)

"""
    range_bottom: which actor to start feature extraction
    range_top: which actor to end feature extraction
    mod : MFCC, MEL ...
"""
def extract_english_features(range_bottom,range_top,mod):
    if not os.path.exists("ExtractedFeatures/english/"):
        os.mkdir("ExtractedFeatures/english/")
    #extract_features_by_speaker("english", "13")
    #extract_features_by_speaker("english","12")
    for i in range(range_bottom,range_top):
        extract_features_by_speaker_with_mod("english",str(i),mod)

"""
    range_bottom: which actor to start feature extraction
    range_top: which actor to end feature extraction
    mod : MFCC, MEL ...
"""
def extract_mandarin_features(range_bottom,range_top,mod):
    if not os.path.exists("ExtractedFeatures/mandarin/"):
        os.mkdir("ExtractedFeatures/mandarin/")
    if range_top == 10:
        extract_features_by_speaker_with_mod("mandarin", str(10),mod)
        for i in range(range_bottom, range_top-1):
            extract_features_by_speaker_with_mod("mandarin", "0" + str(i),mod)
    else:
        for i in range (range_bottom,range_top):
            extract_features_by_speaker_with_mod("mandarin","0"+str(i),mod)

def main():
    #extract_english_features()
    extract_mandarin_features(1,6,"TEMPO")

main()
