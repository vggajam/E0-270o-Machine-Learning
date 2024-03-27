from argparse import ArgumentParser, Namespace

from utils import *
from model import LogisticRegression, SoftmaxRegression
from train_utils import fit_model, evaluate_model


def main(args: Namespace):
    # Get the training data
    X, y = get_data(args.train_data_path, is_binary=(args.mode == "logistic"))
    X_train, y_train, X_val, y_val = train_test_split(X, y)
    
    # Create the model
    model = LogisticRegression(X.shape[1]) if args.mode == "logistic"\
                else SoftmaxRegression(X.shape[1], len(np.unique(y)))

    # Train the model
    train_losses, train_accs, test_losses, test_accs = fit_model(
        model, X_train, y_train, X_val, y_val, num_iters=args.num_iters,
        lr=args.lr, batch_size=args.batch_size, l2_lambda=args.l2_lambda,
        grad_norm_clip=args.grad_norm_clip, is_binary=(args.mode == "logistic"))
    
    # Plot the losses
    plot_losses(train_losses, test_losses, "Losses")

    # Plot the accuracies
    plot_accuracies(train_accs, test_accs, "Accuracies")

    # Get the test data
    X_test, y_test = get_data(args.test_data_path, is_binary=(args.mode == "logistic"))

    # Evaluate the model
    test_loss, test_acc = evaluate_model(
        model, X_test, y_test, args.batch_size, is_binary=(args.mode == "logistic"))
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    # Parse the command line arguments
    parser = ArgumentParser(description='E0-270 (O) Assignment 2')
    parser.add_argument(
        '--train_data_path', type=str, default="data/mnist_train.npz",
        help='Path to the training data')
    parser.add_argument(
        '--test_data_path', type=str, default="data/mnist_test.npz",
        help='Path to the test data')
    parser.add_argument(
        '--mode', type=str, default="logistic", choices=["logistic", "softmax"],
        help='Binary or Multiclass classification mode')
    parser.add_argument(
        '--num_iters', type=int, default=1000,
        help='Number of iterations for training')
    parser.add_argument(
        '--lr', type=float, default=1e-2,
        help='Learning rate for training')
    parser.add_argument(
        '--batch_size', type=int, default=256,
        help='Batch size for training')
    parser.add_argument(
        '--l2_lambda', type=float, default=0.1,
        help='L2 regularization for training')
    parser.add_argument(
        '--grad_norm_clip', type=float, default=4.0,
        help='Clip gradient norm')
    args = parser.parse_args()

    # set the seed
    np.random.seed(2024)

    # Run the main function
    main(args)
