import numpy
import numpy as np
import pandas

import util.CreateDataFrames as cdf

def saveAverageOfFeatureArray(feature_array):
    b = []
    for i in feature_array:
        b.append(np.average(i))
    x = [b]
    return x

def saveFeaturesToDictionary(feature_array):
    b = []
    for i in feature_array:
        b.append(np.average(i))
    x = [b]
    return x

def getMergedDataFrames(speaker_name,language,features):
    matrix = [[]]
    for feature in features:
        tmp = numpy.array(
                saveAverageOfFeatureArray(cdf.createMasterDataFrameAllEmotions(speaker_name, language, feature)))
        tmp = numpy.array(tmp).flatten()
        if feature == features[0]:
            matrix[0] =  tmp
        else:
            matrix.append(tmp)

    ##print(list)
    matrix = pandas.DataFrame(matrix)
    #matrix = np.array(list)
    return matrix.T




    

