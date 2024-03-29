import matplotlib.pyplot as plt
"""
    :argument
    y : Target Vector
    X : Predicted Values
"""
def printScores(y_test,y_pred):
    scores = [0, 0, 0, 0, 0]
    for j in range(0, len(y_pred)):
        if y_test[j] == y_pred[j] and y_test[j] == "Angry":
            scores[0] = scores[0] + 1
        elif y_test[j] == y_pred[j] and y_test[j] == "Happy":
            scores[1] = scores[1] + 1
        elif y_test[j] == y_pred[j] and y_test[j] == "Neutral":
            scores[2] = scores[2] + 1
        elif y_test[j] == y_pred[j] and y_test[j] == "Sad":
            scores[3] = scores[3] + 1
        elif y_test[j] == y_pred[j] and y_test[j] == "Surprise":
            scores[4] = scores[4] + 1
    count = [0, 0, 0, 0, 0]
    for j in range(0, len(y_pred)):
        if y_test[j] == "Angry":
            count[0] += 1
        elif y_test[j] == "Happy":
            count[1] += 1
        elif y_test[j] == "Neutral":
            count[2] += 1
        elif y_test[j] == "Sad":
            count[3] += 1
        elif y_test[j] == "Surprise":
            count[4] += 1

    if count[0] != 0:
        print("angry score:", scores[0]/count[0] * 100)
    if count[1] != 0:
        print("happy score:", scores[1]/count[1] * 100)
    if count[2] != 0:
        print("neutral score:", scores[2] /count[2] * 100)
    if count[3] != 0:
        print("sad score:", scores[3]/count[3] * 100)
    if count[4] != 0:
        print("surprise score:", scores[4]/count[4] * 100)

        

def printScoresOld(y_test,y_pred):
    scores = [0, 0, 0, 0, 0]
    for j in range(0, len(y_pred)):
        if y_test[j] == y_pred[j] and y_test[j] == "Angry":
            scores[0] = scores[0] + 1
        elif y_test[j] == y_pred[j] and y_test[j] == "Happy":
            scores[1] = scores[1] + 1
        elif y_test[j] == y_pred[j] and y_test[j] == "Neutral":
            scores[2] = scores[2] + 1
        elif y_test[j] == y_pred[j] and y_test[j] == "Sad":
            scores[3] = scores[3] + 1
        elif y_test[j] == y_pred[j] and y_test[j] == "Surprise":
            scores[4] = scores[4] + 1
    count = [0, 0, 0, 0, 0]
    for j in range(0, len(y_pred)):
        if y_test[j] == "Angry":
            count[0] += 1
        elif y_test[j] == "Happy":
            count[1] += 1
        elif y_test[j] == "Neutral":
            count[2] += 1
        elif y_test[j] == "Sad":
            count[3] += 1
        elif y_test[j] == "Surprise":
            count[4] += 1

    print("angry score:", scores[0] / count[0] * 100)

    print("happy score:", scores[1] / count[1] * 100)

    print("neutral score:", scores[2] / count[2] * 100)

    print("sad score:", scores[3] / count[3] * 100)

    print("surprise score:", scores[4] / count[4] * 100)

def printScoresAsNegPos(y_test,y_pred):
    scores = [0, 0, 0, 0, 0]
    for j in range(0, len(y_pred)):
        if y_test[j] == y_pred[j] and y_test[j] == "Angry":
            scores[0] = scores[0] + 1
        elif y_test[j] == y_pred[j] and y_test[j] == "Happy":
            scores[1] = scores[1] + 1
        elif y_test[j] == y_pred[j] and y_test[j] == "Neutral":
            scores[2] = scores[2] + 1
        elif y_test[j] == y_pred[j] and y_test[j] == "Sad":
            scores[3] = scores[3] + 1
        elif y_test[j] == y_pred[j] and y_test[j] == "Surprise":
            scores[4] = scores[4] + 1
    count = [0, 0, 0, 0, 0]
    for j in range(0, len(y_pred)):
        if y_test[j] == "Angry":
            count[0] += 1
        elif y_test[j] == "Happy":
            count[1] += 1
        elif y_test[j] == "Neutral":
            count[2] += 1
        elif y_test[j] == "Sad":
            count[3] += 1
        elif y_test[j] == "Surprise":
            count[4] += 1

    print("positive score: ", (scores[1]/count[1] + scores[4]/count[4]) * 100)

    print("neutral score: ", scores[2] / count[2] * 100)

    print("negative score: ", (scores[0]/count[0] + scores[3]/count[3]) * 100)

