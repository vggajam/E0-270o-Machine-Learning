from utils import load_data
from model import GaussianNaiveBayes


def main() -> None:
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Create model
    model = GaussianNaiveBayes()

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_train)
    accuracy = (y_pred == y_train).mean()
    print(f"Train accuracy: {accuracy:.4f}")

    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"Test Accuracy: {accuracy:.3f}")

    # Print the confusion matrix 
    #   and relevant metrics for each class
    #########################################
    # YOUR CODE HERE
    #########################################
    # 
    # 
    #########################################
    
if __name__ == "__main__":
    main()
