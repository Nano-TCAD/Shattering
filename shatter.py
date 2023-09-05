import numpy as np

from grid import Grid
from numpy import typing as npt
from typing import Tuple

Matrix = npt.NDArray[np.complexfloating]


def shatter(A: Matrix, gamma: float) -> Tuple[Matrix, Grid, float]:
    
    # Validate matrix A
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]

    # Complex Ginibre matrix
    rng = np.random.default_rng()
    mean = 1 / n
    G = rng.normal(loc=mean, scale=1, size=(n, n)) + 1j * rng.normal(loc=mean, scale=1, size=(n, n))

    X = A + gamma * G  # What about E?
    g = Grid(n, gamma)
    epsilon = 0.5 * gamma ** 5 / (16 * n ** 9)

    return X, g, epsilon


if __name__ == "__main__":

    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.complex128)
    gamma = 0.5
    X, g, epsilon = shatter(A, gamma)
    print(X)
    print(g)
    print(epsilon)
