import numpy as np

import CreateDataFrames as cdf

def getMergedDataFrames(speaker_name,language,features):
    dataframes =[]
    for feature in features:
        np.append(dataframes,cdf.createMasterDataFrameAllEmotions(speaker_name,language,feature))

    for dataframe in dataframes:
        shape = dataframe.shape
        print(shape)
        for i in range(shape):
            dataframe = np.average(dataframe[i])


getMergedDataFrames("11","english",["MFCC","MEL"])

    

