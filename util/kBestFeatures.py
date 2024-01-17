import sklearn.feature_selection as skl

def SelectKBestFeatures(vector,target_vector,K):
    best_features = skl.SelectKBest(skl.f_classif, k=K)
    new_features = best_features.fit_transform(X=vector, y=target_vector)
    return new_features
