


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

def createTargetVectorALL(SIZE):
    return createTargetVector(SIZE,True,True,True,True,True)