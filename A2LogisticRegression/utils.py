import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


def get_data(
        data_path: str, is_binary: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load the data from the given path and return the features and labels.

    Args:
        data_path: str, path to the .npz file containing the data
        is_binary: bool, if True, only load the data for classes 0 and 1
    
    Returns:
        X: np.ndarray, features
        y: np.ndarray, labels
    '''
    data = np.load(data_path)
    if 'train' in data_path:
        X, y = data['train_images'], data['train_labels']
    else:
        X, y = data['test_images'], data['test_labels']
    if is_binary:
        idxs0 = np.where(y == 0)[0]
        idxs1 = np.where(y == 1)[0]
        idxs = np.concatenate([idxs0, idxs1])
        X = X[idxs]
        y = y[idxs]
    X = X.reshape(X.shape[0], -1)
    X = X / 127.5 - 1
    return X, y


def train_test_split(
        X: np.ndarray, y: np.ndarray, test_ratio: int = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Split the data into training and validation sets.

    Args:
        X: np.ndarray, features
        y: np.ndarray, labels
        test_ratio: float, ratio of the data to be used for validation

    Returns:
        X_train: np.ndarray, features for training
        y_train: np.ndarray, labels for training
        X_val: np.ndarray, features for validation
        y_val: np.ndarray, labels for validation
    '''
    assert test_ratio < 1 and test_ratio > 0
    idxs = np.random.permutation(X.shape[0])
    X, y = X[idxs], y[idxs]
    split_idx = int(X.shape[0] * (1 - test_ratio))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def get_data_batch(
        X: np.ndarray, y: np.ndarray, batch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Get a batch of the given dataset.

    Args:
    - X: np.ndarray, images
    - y: np.ndarray, labels
    - batch_size: int, size of the batch

    Returns:
    - X_batch: np.ndarray, batch of images
    - y_batch: np.ndarray, batch of labels
    '''
    idxs = # TODO: get random indices of size batch_size from the dataset without replacement
    return X[idxs], y[idxs]


def plot_losses(
        train_losses: list, val_losses: list, title: str
) -> None:
    '''
    Plot the training and validation losses.

    Args:
        train_losses: list, training losses
        val_losses: list, validation losses
        title: str, title of the plot
    '''
    raise NotImplementedError("Plot the training and validation losses.")
    plt.savefig('loss.png')


def plot_accuracies(
        train_accs: list, val_accs: list, title: str
) -> None:
    '''
    Plot the training and validation accuracies.

    Args:
        train_accs: list, training accuracies
        val_accs: list, validation accuracies
        title: str, title of the plot
    '''
    raise NotImplementedError("Plot the training and validation accuracies.")
    plt.savefig('acc.png')
