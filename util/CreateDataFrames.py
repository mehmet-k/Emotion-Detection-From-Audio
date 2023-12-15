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
        
        mod: name of the mod, for example: MFCC, MEL
             to save standart extracted features, send an empty string = ""
             let's use all upper case naming just for compatibility
    
    :return
        np array of extract spedified features of specified emotions
"""
def createMasterDataFrame(speaker_name,language,mod,ANGRY,HAPPY,NEUTRAL,SAD,SURPRISE):
    if int(speaker_name) >= 10:
        os.chdir("ExtractedFeatures/" + language + "/00" + speaker_name)
    else:
        os.chdir("ExtractedFeatures/" + language + "/000" + speaker_name)

    master_data_frame = []
    if ANGRY:
        data_angry = pd.read_csv('Angry'+mod+'.csv')
        master_data_frame = data_angry
    if HAPPY:
        data_happy = pd.read_csv('Happy'+mod+'.csv')
        if not ANGRY:
            master_data_frame = data_happy
        else:
            master_data_frame = np.concatenate((master_data_frame, data_happy))
    if NEUTRAL:
        data_neutral = pd.read_csv('Neutral'+mod+'.csv')
        if not HAPPY:
            master_data_frame = data_neutral
        else:
            master_data_frame = np.concatenate((master_data_frame, data_neutral))
    if SAD:
        data_sad = pd.read_csv('Sad'+mod+'.csv')
        if not NEUTRAL:
            master_data_frame = data_sad
        else:
            master_data_frame = np.concatenate((master_data_frame, data_sad))
    if SURPRISE:
        data_surprise = pd.read_csv('Surprise'+mod+'.csv')
        if not SURPRISE:
            master_data_frame = data_surprise
        else:
            master_data_frame = np.concatenate((master_data_frame, data_surprise))

    os.chdir("../../..")
    return master_data_frame

"""
    Extract MFCC features of all emotions by speaker,language
    
    :argument
    
        speaker_name: name of the actor/actress (format: 1,2,3..19)
    
        language: mandarin or english
    
        ANGRY: true , include angry
            false, ignore angry
            (rest of the parameters are same logic)
    
    :return
    
        np array of extracted specified features of all emotions
"""
def createMasterDataFrameAllEmotions(speaker_name,language,mod):
    return createMasterDataFrame(speaker_name,language,mod,True,True,True,True,True)