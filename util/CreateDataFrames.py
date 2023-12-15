import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.feature_selection as skl
import pandas as pd

"""
    Extract MFCC features of specified emotions by speaker,language
    :argument
        speaker_name: name of the actor/actress (format: 1,2,3..19)
        language: mandarin or english
    ANGRY: true , include angry
           false, ignore angry
           (rest of the parameters are same logic)
    :return
        np array of extract MFCC features of specified emotions
"""
def createMasterDataFrameMFCC(speaker_name,language,ANGRY,HAPPY,NEUTRAL,SAD,SURPRISE):
    if int(speaker_name) >= 10:
        os.chdir("ExtractedFeatures/" + language + "/00" + speaker_name)
    else:
        os.chdir("ExtractedFeatures/" + language + "/000" + speaker_name)

    master_data_frame = []
    if ANGRY:
        data_angry = pd.read_csv('AngryMFCC.csv')
        master_data_frame = data_angry
    if HAPPY:
        data_happy = pd.read_csv('HappyMFCC.csv')
        if not ANGRY:
            master_data_frame = data_happy
        else:
            master_data_frame = np.concatenate((master_data_frame, data_happy))
    if NEUTRAL:
        data_neutral = pd.read_csv('NeutralMFCC.csv')
        if not HAPPY:
            master_data_frame = data_neutral
        else:
            master_data_frame = np.concatenate((master_data_frame, data_neutral))
    if SAD:
        data_sad = pd.read_csv('SadMFCC.csv')
        if not NEUTRAL:
            master_data_frame = data_sad
        else:
            master_data_frame = np.concatenate((master_data_frame, data_sad))
    if SURPRISE:
        data_surprise = pd.read_csv('SurpriseMFCC.csv')
        if not SURPRISE:
            master_data_frame = data_surprise
        else:
            master_data_frame = np.concatenate((master_data_frame, data_surprise))

    os.chdir("../../..")
    return master_data_frame


#NOT IMPLEMENTED YET
def createMasterDataFrameALL(speaker_name,language,ANGRY,HAPPY,NEUTRAL,SAD,SURPRISE):
    if int(speaker_name) >= 10:
        os.chdir("ExtractedFeatures/"+language+"/00" + speaker_name)
    else:
        os.chdir("ExtractedFeatures/" + language + "/000" + speaker_name)
    master_data_frame = []
    if ANGRY:
        data_angry = pd.read_csv('Angry.csv')
        master_data_frame = np.concatenate((master_data_frame,data_angry))
    elif HAPPY:
        data_happy = pd.read_csv('Happy.csv')
        master_data_frame = np.concatenate((master_data_frame, data_happy))
    elif NEUTRAL:
        data_neutral = pd.read_csv('Neutral.csv')
        master_data_frame = np.concatenate((master_data_frame, data_neutral))
    elif SAD:
        data_sad = pd.read_csv('Sad.csv')
        master_data_frame = np.concatenate((master_data_frame, data_sad))
    elif SURPRISE:
        data_surprise = pd.read_csv('Surprise.csv')
        master_data_frame = np.concatenate((master_data_frame, data_surprise))

    os.chdir("../../..")
    return master_data_frame


#NOT IMPLEMENTED YET
def createMasterDataFrameMEL(speaker_name,language,ANGRY,HAPPY,NEUTRAL,SAD,SURPRISE):
    os.chdir("ExtractedFeatures/+"+language+"/00" + speaker_name)
    master_data_frame = []
    if ANGRY:
        data_angry = pd.read_csv('AngryMELcsv')
        master_data_frame = np.concatenate((master_data_frame, data_angry))
    elif HAPPY:
        data_happy = pd.read_csv('HappyMEL.csv')
        master_data_frame = np.concatenate((master_data_frame, data_happy))
    elif NEUTRAL:
        data_neutral = pd.read_csv('NeutralMEL.csv')
        master_data_frame = np.concatenate((master_data_frame, data_neutral))
    elif SAD:
        data_sad = pd.read_csv('SadMEL.csv')
        master_data_frame = np.concatenate((master_data_frame, data_sad))
    elif SURPRISE:
        data_surprise = pd.read_csv('SurpriseMEL.csv')
        master_data_frame = np.concatenate((master_data_frame, data_surprise))

    os.chdir("../../..")
    return master_data_frame