"""Least squares (minimum-norm) and ridge regression, implemented with numpy only.

Notation follows the course (slides "9. linear", p. 4-5):
    X     design matrix, shape (m, d): m training examples, d features
    y     labels, shape (m,)
    w     weight vector, shape (d,)
    alpha ridge regularization parameter
"""
import numpy as np


def fit_least_squares(X, y):
    """Least squares solution w_S = argmin ||Xw - y||^2.

    For d <= m (X full rank) this is (X^T X)^{-1} X^T y, as in the slides.
    For d > m there are infinitely many solutions: we return the one with
    minimum norm, X^+ y (pseudo-inverse). A single formula covers both cases.
    """
    # TODO 1: return the minimum-norm least squares solution.
    # Hint: one line, using np.linalg.pinv (see theory doc, Section 4.4).
    w = np.linalg.pinv(X) @ y #done
    return w
    



def fit_ridge(X, y, alpha):
    """Ridge regression w = (alpha I + X^T X)^{-1} X^T y (slides 9, p. 5)."""
    # TODO 2: solve the linear system (alpha I + X^T X) w = X^T y.
    # Hint: build the d x d matrix with X.T @ X and np.eye(d),
    # then use np.linalg.solve instead of computing the inverse.
    d = X.shape[1]
    b = X.T @ y
    A = alpha * np.eye(d) + X.T @ X
    w = np.linalg.solve(A, b)  #done
    return w


def mse(X, y, w):
    """Mean squared error (1/m) * ||Xw - y||^2: training or test error."""
    # TODO 3: compute the predictions X @ w and return the mean squared error.
    y_hat = X @ w
    error = np.mean((y_hat - y) ** 2)
    return error


if __name__ == "__main__":
    # Self-checks from the theory doc. Run: python src/models.py

    # Check A (theory doc, Section 4.5): one example x = (1, 1), y = 2.
    # Infinitely many solutions; the minimum-norm one is w = (1, 1).
    X = np.array([[1.0, 1.0]])
    y = np.array([2.0])
    w = fit_least_squares(X, y)
    print("Check A, expected [1. 1.]:", w)

    # Check B (theory doc, Section 5.1): rows (1, 0) and (1, eps), noise delta.
    # Expected w = (1, delta / eps): a small noise gives a huge weight.
    eps, delta = 0.01, 0.1
    X = np.array([[1.0, 0.0], [1.0, eps]])
    y = np.array([1.0, 1.0 + delta])
    w = fit_least_squares(X, y)
    print("Check B, expected [1. 10.]:", w)

    # Check C: ridge on the same data keeps the weights small.
    w_ridge = fit_ridge(X, y, alpha=0.1)
    print("Check C, ridge weights (should be much smaller than 10):", w_ridge)

    # Check D: the least squares fit interpolates, so its training error is 0.
    print("Check D, expected ~0:", mse(X, y, w))
