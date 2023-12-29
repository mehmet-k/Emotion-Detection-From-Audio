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
def save_audio_to_array(language,speaker_name,category):
    audio = []
    for file in os.listdir("trainingSet/"+language+"/00"+speaker_name+ "/" + category):
        audio.append("trainingSet/"+language+"/00"+speaker_name+ "/" + category+"/"+file)

    return audio

def saveFeaturesToDictionary(feature_name,feature_array,extracted_features):
    b = []
    for i in feature_array:
        b.append(np.average(i))
    x = [np.average(b)]
    extracted_features.update({feature_name: x})

def saveFlattenedFeaturesToDictionary(feature_array):
    b = []
    for i in feature_array:
        b.append(np.average(i))
    x = [b]
    return x

def extractOnlyMEL(language,speaker_name, category,audio):
    y, sr = librosa.load(audio)
    # save name to write a file
    os.chdir("ExtractedFeatures/" + language + "/00" + speaker_name + "/")
    #####################FEATURE EXTRACTIONS#####################
    # MEL
    a = librosa.feature.melspectrogram(y=y, sr=sr)
    a = saveFlattenedFeaturesToDictionary(a)

    df = pd.DataFrame(a)
    df.to_csv(category + 'MEL.csv', mode='a', header=False, index=False)
    # to be completed
    print("CATEGORY: ", category, " from: ", audio, " features has been saved")
    os.chdir("../../..")
def extractOnlyMFCC(language,speaker_name, category,audio):
    y, sr = librosa.load(audio)
    # save name to write a file
    os.chdir("ExtractedFeatures/" + language + "/00" + speaker_name + "/")
    #####################FEATURE EXTRACTIONS#####################
    # MFCC
    a = librosa.feature.mfcc(y=y, sr=sr)
    a = saveFlattenedFeaturesToDictionary( a)

    df = pd.DataFrame(a)
    df.to_csv(category + 'MFCC.csv', mode='a', header=False, index=False)
    # to be completed
    print("CATEGORY: ", category, " from: ", audio, " features has been saved")
    os.chdir("../../..")

#extract features from specified file
#then write feature values to a file
#NOT COMPLETE, DEMO VERSION
def extract_features_by_file(language,speaker_name, category,audio,extracted_features):
    #load audio, save audio file as floating point numbers to y
    y, sr = librosa.load(audio)
    # save name to write a file
    os.chdir("ExtractedFeatures/"+language+"/00" + speaker_name + "/")
    #####################FEATURE EXTRACTIONS#####################
    #MFCC
    a = librosa.feature.mfcc(y=y, sr=sr)
    saveFeaturesToDictionary('mfcc',a,extracted_features)

    a = librosa.feature.chroma_stft(y=y,sr=sr)
    saveFeaturesToDictionary('chroma_stft',a,extracted_features)

    a = librosa.feature.chroma_cqt(y=y,sr=sr,bins_per_octave=12,)
    saveFeaturesToDictionary('chroma_cqt',a,extracted_features)

    a=librosa.feature.chroma_vqt(y=y,sr=sr,intervals='ji3')
    saveFeaturesToDictionary('chroma_vqt',a,extracted_features)

    a=librosa.feature.melspectrogram(y=y,sr=sr)
    saveFeaturesToDictionary('melspectrogram',a,extracted_features)

    df = pd.DataFrame(extracted_features)
    df.to_csv(category + '.csv', mode='a', header=False,index=False)
    #to be completed
    print("CATEGORY: ",category," from: ",audio," features has been saved")
    os.chdir("../../..")

#extract features based on emotion category
#category = folder name
def extract_features_by_category(language,speaker_name,category):
    extracted_features = {
        'mfcc': [],
        'chroma_stft': [],
        'chroma_cqt': [],
        'chroma_vqt': [],
        'melspectrogram': []
    }
#    os.chdir("ExtractedFeatures/"+language+"/00" + speaker_name + "/")
#    df = pd.DataFrame(extracted_features)
#    df.to_csv(category + '.csv',index=False)
#    os.chdir("../../..")
    os.chdir("ExtractedFeatures/" + language + "/00" + speaker_name + "/")
    df = pd.DataFrame()
    #df.to_csv(category + 'MFCC.csv', index=False)
    os.chdir("../../..")
    audios = save_audio_to_array(language,speaker_name, category)
    for audio in audios:
        #extract_features_by_file(language,speaker_name,category,audio,extracted_features)
        #extractOnlyMFCC(language,speaker_name,category,audio)
        extractOnlyMEL(language,speaker_name,category,audio)
        
#extract features of an actor/actress
#speaker_name = folder name (0011,0012...)
def extract_features_by_speaker(language,speaker_name):
    if not os.path.exists("ExtractedFeatures/"+language+"/00" + speaker_name):
        os.mkdir("ExtractedFeatures/"+language+"/00" + speaker_name)
    for category in emotion_categories:
        extract_features_by_category(language,speaker_name,category)

#extract features from english speaking training set
def extract_english_features():
    if not os.path.exists("ExtractedFeatures/english/"):
        os.mkdir("ExtractedFeatures/english/")
    #extract_features_by_speaker("english", "13")
    #extract_features_by_speaker("english","12")
    for i in range(11,16):
        extract_features_by_speaker("english",str(i))

def extract_mandarin_features():
    if not os.path.exists("ExtractedFeatures/mandarin/"):
        os.mkdir("ExtractedFeatures/mandarin/")
    for i in range (1,9):
        extract_features_by_speaker("mandarin","0"+str(i))
    extract_features_by_speaker("mandarin",str(10))
def main():
    extract_english_features()
    #extract_mandarin_features()

main()
