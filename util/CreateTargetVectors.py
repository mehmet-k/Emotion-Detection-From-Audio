

"""
    :argument
    SIZE: size of single emotion category, for e.g. if size of data frame with all emotions is k,
          size should be len(dataframe)/k
    ANGRY: true , include angry
           false, ignore angry
        (rest of the parameters are same logic)
    :return
    target vector
"""
def createTargetVector(SIZE,ANGRY,HAPPY,NEUTRAL,SAD,SURPRISE):
    vector = []
    SIZE = int(SIZE)
    if ANGRY:
        for i in range(0,SIZE):
            vector.append('Angry')
    if HAPPY:
        for i in range(0,SIZE):
            vector.append('Happy')
    if NEUTRAL:
        for i in range(0,SIZE):
            vector.append('Neutral')
    if SAD:
        for i in range(0,SIZE):
            vector.append('Sad')
    if SURPRISE:
        for i in range(0,SIZE):
            vector.append('Surprise')
    return vector

"""
    :argument
    SIZE: size of single emotion category, for e.g. if size of data frame with all emotions is k,
          size should be len(dataframe)/k
    call this if you want to include all emotions
    :return
    target vector
"""
def createTargetVectorALL(SIZE):
    return createTargetVector(SIZE,True,True,True,True,True)