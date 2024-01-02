import numpy as np

import CreateDataFrames as CDF

def getMergedDataFrames(speaker_name,language,features):
    dataframes =[]
    for feature in features:
        dataframes.append(CDF.createMasterDataFrameAllEmotions(speaker_name,language,feature))

    for dataframe in dataframes:
        shape = dataframe.shape
        for i in range(shape):
            dataframe[i] = np.average(i)

    

