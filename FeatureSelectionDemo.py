import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn.feature_selection as skl
import pandas as pd


def createTargetVector():
    vector = []
    for i in range(350):
        vector.append('Angry')
    for i in range(350):
        vector.append('Happy')
    for i in range(350):
        vector.append('Neutral')
    for i in range(350):
        vector.append('Sad')
    for i in range(350):
        vector.append('Suprise')
    return vector

def createMasterDataFrame(speaker_name):
    os.chdir("ExtractedFeatures/english/00"+speaker_name)
    data_angry = pd.read_csv('Angry.csv')
    data_happy = pd.read_csv('Happy.csv')
    data_neutral = pd.read_csv('Neutral.csv')
    data_sad = pd.read_csv('Sad.csv')
    data_surprise = pd.read_csv('Surprise.csv')
    master_data_frame = [data_angry,data_happy,data_neutral,data_sad,data_surprise]
    master_data_frame  = pd.concat(master_data_frame )
    os.chdir("../../..")
    return master_data_frame

def SelectKBest_Data_Frame(data_frame,target_vector):
    best_features = skl.SelectKBest(skl.f_classif, k=2)
    new_features = best_features.fit_transform(X=data_frame, y=target_vector)
    new_data_frame = pd.DataFrame(data=new_features, columns=best_features.get_feature_names_out())
    return new_data_frame

masterDF = createMasterDataFrame('11')
y=createTargetVector()

f_statistic,p_values = skl.f_classif(X=masterDF,y=y)
print("f_statistics: ", f_statistic)
print("p_values: ", p_values)

if not os.path.exists("K_best_features"):
    os.mkdir("K_best_features")

os.chdir("K_best_features")
masterDF.to_csv("before.csv",index=False)
new_master_df = SelectKBest_Data_Frame(masterDF,y)
new_master_df.to_csv("after.csv",index=False)
os.chdir('../')