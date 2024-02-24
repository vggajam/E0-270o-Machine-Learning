from utils import load_data
from model import GaussianNaiveBayes
import numpy as np

def get_conf_mat(no_of_labels, y_truth, y_pred) -> np.ndarray:
    conf_matrix = np.zeros((no_of_labels, no_of_labels), dtype=np.int32)
    for truth, pred in zip(y_truth, y_pred):
        conf_matrix[truth][pred] += 1
    return conf_matrix

def get_precision(conf_mat, label_idx):
    true_positives = conf_mat[label_idx, label_idx]
    false_positives = np.sum(conf_mat[:, label_idx]) - true_positives
    precision_value = true_positives / (true_positives + false_positives)
    return precision_value

def get_recall(conf_mat, label_idx):
    true_positives = conf_mat[label_idx, label_idx]
    false_negatives = np.sum(conf_mat[label_idx, :]) - true_positives
    recall_value = true_positives / (true_positives + false_negatives)
    return recall_value

def get_f1_score(conf_mat, label_idx):
    prec = get_precision(conf_mat, label_idx)
    rec = get_recall(conf_mat, label_idx)
    f1_score_value = 2 * (prec * rec) / (prec + rec)
    return f1_score_value

def main() -> None:
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Create model
    model = GaussianNaiveBayes()

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_train_pred = model.predict(X_train)
    accuracy = (y_train_pred == y_train).mean()
    print(f"Train accuracy: {accuracy:.4f}")

    y_test_pred = model.predict(X_test)
    accuracy = (y_test_pred == y_test).mean()
    print(f"Test Accuracy: {accuracy:.3f}")

    # Print the confusion matrix 
    #   and relevant metrics for each class
    #########################################
    # YOUR CODE HERE
    #########################################
    LABEL_COUNT = 10
    print("Train Data metrics:")
    print("-------------------")
    print("Confusion Matrix:")
    confusion_matrix = get_conf_mat(LABEL_COUNT, y_train, y_train_pred)
    print(confusion_matrix)
    print("Precision:")
    for label in range(LABEL_COUNT):
        print(f"{get_precision(confusion_matrix, label)} ", end="")
    print("")
    print("Recall:")
    for label in range(LABEL_COUNT):
        print(f"{get_recall(confusion_matrix, label)} ", end="")
    print("")
    print("F1 score:")
    for label in range(LABEL_COUNT):
        print(f"{get_f1_score(confusion_matrix, label)} ", end="")
    print("")

    print("Test Data metrics:")
    print("-------------------")
    print("Confusion Matrix:")
    confusion_matrix = get_conf_mat(LABEL_COUNT, y_test, y_test_pred)
    print(confusion_matrix)
    print("Precision:")
    for label in range(LABEL_COUNT):
        print(f"{get_precision(confusion_matrix, label)} ", end="")
    print("")
    print("Recall:")
    for label in range(LABEL_COUNT):
        print(f"{get_recall(confusion_matrix, label)} ", end="")
    print("")
    print("F1 score:")
    for label in range(LABEL_COUNT):
        print(f"{get_f1_score(confusion_matrix, label)} ", end="")
    print("")
    #########################################
    
if __name__ == "__main__":
    main()
