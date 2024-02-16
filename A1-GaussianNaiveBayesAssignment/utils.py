import numpy as np
from typing import Tuple
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_data(
        test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    digits = load_digits()
    X, y = digits.data, digits.target
    return train_test_split(X, y, test_size=test_size, random_state=2023)
