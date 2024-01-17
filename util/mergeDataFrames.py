import numpy
import numpy as np
import pandas

import sklearn.feature_selection as skl
import util.CreateDataFrames as cdf
import util.CreateTargetVectors as cTV

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

def getAveragedMergedDataFrames(speaker_name,language,features):
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

def getMergedDataFrames(speaker_name,language,features):
    List = []
    for feature in features:
        List.append(numpy.array(cdf.createMasterDataFrameAllEmotions(speaker_name, language, feature)))

    list_of_lists = []
    for i in range(0, 1750):
        tmp = []
        for feature in range(len(features)):
            tmp.append(List[feature][i])
        tmp = numpy.hstack(tmp)
        list_of_lists.append(tmp)

    ##print(list)
    #matrix = pandas.DataFrame(list_of_lists).T
    list_of_lists = np.array(list_of_lists)
    return list_of_lists
def SelectKBest_Data_Frame(data_frame,target_vector,K):
    best_features = skl.SelectKBest(skl.f_classif, k=K)
    new_features = best_features.fit_transform(X=data_frame, y=target_vector)
    new_data_frame = pandas.DataFrame(data=new_features, columns=best_features.get_feature_names_out())
    return new_data_frame

def getMergedDataFramesWithKBestFeatures(speaker_name,language,features,k):
    matrix = [[]]
    for feature in features:
        data_frame = cdf.createMasterDataFrameAllEmotions(speaker_name, language, feature)
        target_vector = cTV.createTargetVectorALL(350)
        tmp = numpy.array(SelectKBest_Data_Frame(data_frame,target_vector,k))
        tmp = numpy.array(tmp).flatten()
        if feature == features[0]:
            matrix[0] = tmp
        else:
            matrix.append(tmp)

    list_of_lists = []
    lower=0
    x=0

    for i in range(0,1750):
        list = []
        for feature in range(len(features)):
            for j in range(lower+x,k+x):
                list.append(matrix[feature][j])
        x = x + len(features)
        #print(len(list))
        #np.append(new_matrix,list)
        list_of_lists.append(list)
        #np.column_stack(new_matrix,list)

    ##print(list)
    new_matrix = np.array(list_of_lists)
    new_matrix = pandas.DataFrame(new_matrix)
    # matrix = np.array(list)
    return new_matrix







