from utils import load_data
from model import GaussianNaiveBayes
import numpy as np

CLASS_COUNT = 10

def get_conf_mat(y_truth, y_pred) -> np.ndarray:
    conf_matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int32)
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

def print_metrics(y_given, y_pred):
    
    print("Confusion Matrix:")
    confusion_matrix = get_conf_mat(y_given, y_pred)
    print(confusion_matrix)
    print("Precision:")
    for label in range(CLASS_COUNT):
        print(f"{get_precision(confusion_matrix, label):.5f} ", end="")
    print("")
    print("Recall:")
    for label in range(CLASS_COUNT):
        print(f"{get_recall(confusion_matrix, label):.5f} ", end="")
    print("")
    print("F1 score:")
    for label in range(CLASS_COUNT):
        print(f"{get_f1_score(confusion_matrix, label):.5f} ", end="")
    print("")

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
    print(f"-------------------\nTrain Data metrics:\n-------------------")
    print_metrics(y_train, y_train_pred)
    print("-------------------\n")
    print(f"-------------------\nTest Data metrics:\n-------------------")
    print_metrics(y_test, y_test_pred)
    print("-------------------\n")
    #########################################
    
if __name__ == "__main__":
    main()
