from sklearn.metrics import confusion_matrix


def calculate_accuracy(conf_matrix, class_labels):
    accuracies = []

    for i in range(len(class_labels)):
        correct_predictions = conf_matrix[i, i]
        total_predictions = np.sum(conf_matrix[i, :])
        accuracy = correct_predictions / total_predictions if total_predictions != 0 else 0
        accuracies.append(accuracy)

    return accuracies


def create_confusionMatrix(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

def main(y_test, y_pred):
    class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_accuracies = calculate_accuracy(conf_matrix, class_labels)

    for i in range(len(class_labels)):
        print(f"Accuracy for class {class_labels[i]}: {class_accuracies[i]:.2%}")



