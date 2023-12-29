import matplotlib.pyplot as plt
"""
    :argument
    y : Target Vector
    X : Predicted Values
"""
def printScores(y,X):

    scores = [0,0,0,0,0]

    for j in range(0, len(y)):
        if y[j] == X[j] and y[j] == "Angry":
            scores[0] = scores[0] + 1
        elif y[j] == X[j] and y[j] == "Happy":
            scores[1] = scores[1] + 1
        elif y[j] == X[j] and y[j] == "Neutral":
            scores[2] = scores[2] + 1
        elif y[j] == X[j] and y[j] == "Sad":
            scores[3] = scores[3] + 1
        elif y[j] == X[j] and y[j] == "Surprise":
            scores[4] = scores[4] + 1

    count = [0,0,0,0,0]
    for j in range(0, len(y)):
        if y[j] == "Angry":
            count[0] += 1
        elif y[j] == "Happy":
            count[1] += 1
        elif y[j] == "Neutral":
            count[2] += 1
        elif y[j] == "Sad":
            count[3] += 1
        elif y[j] == "Surprise":
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

        


    emotions = ['angry', 'happy', 'neutral', 'sad', 'surprise']
    scores = [scores[0]/count[0]*100, scores[1]/count[1]*100, scores[2]/count[2]*100, scores[3]/count[3]*100, scores[4]/count[4]*100]

    plt.bar(emotions, scores, color=['#FFB6C1', '#FFD700', '#98FB98', '#ADD8E6', '#FFA07A'])
    plt.xlabel('Emotions', fontweight='bold')
    plt.ylabel('Scores', fontweight='bold')
    plt.title('Emotion Scores')
    plt.ylim(0, 100)  

    plt.show()      


