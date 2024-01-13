import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def calculateResults(conf_matrix, class_labels):

    results = {}
    class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
    
    for i, class_label in enumerate(class_labels):
        # True positives, false positives, true negatives, and false negatives
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[i, :]) - tp
        tn = np.sum(np.diag(conf_matrix)) - (tp + np.sum(conf_matrix[:, i]))
        fn = np.sum(conf_matrix[:, i]) - tp

        # Calculate accuracy, recall, and precision
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0

        results[class_label] = {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision
        }
    for class_label, metrics in results.items():
        print(f"Class: {class_label}")
        print(f"  Accuracy: {metrics['accuracy']:.2f}")
        print(f"  Recall: {metrics['recall']:.2f}")
        print(f"  Precision: {metrics['precision']:.2f}")
        f1score = 2*(metrics['recall']*metrics['precision']/(metrics['recall'] + metrics['precision']))
        print("F1 Score: ", f1score)
        print("-" * 20)  

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    return results


    
