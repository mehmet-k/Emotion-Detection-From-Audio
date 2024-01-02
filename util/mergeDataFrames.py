import numpy as np

import util.CreateDataFrames as cdf

def saveAverageOfFeatureArray(feature_array):
    b = []
    for i in feature_array:
        b.append(np.average(i))
    x = [np.average(b)]
    return x
def getMergedDataFrames(speaker_name,language,features):
    dataframes = [[]]
    dataframes[0] = cdf.createMasterDataFrameAllEmotions(speaker_name,language,features[0])
    dataframes.append(cdf.createMasterDataFrameAllEmotions(speaker_name,language,features[1]))
    #for feature in range(1,len(features)):
    #    np.append(dataframes,cdf.createMasterDataFrameAllEmotions(speaker_name,language,features[feature]))
    print(len(dataframes[0]),len(dataframes[0][0]))
    print(len(dataframes[1]), len(dataframes[1][0]))

    tmp_dataframe = [[]]
    for dataframe in dataframes:
        tmp_array = []
        shape = dataframe.shape
        print("before:",shape)
        for i in range(len(dataframe)):
            tmp_array.append(saveAverageOfFeatureArray(dataframe[i]))
        np.concatenate((tmp_dataframe,tmp_array))
        print("after:" ,len(tmp_dataframe))



    

