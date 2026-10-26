"""Experiment: train and test error as a function of the number of features d (synthetic data).

For each d we fit the model on the first d features and measure the errors.
We repeat everything on n_seeds independent datasets and take the average,
so that the curve does not depend on one lucky or unlucky draw.
"""
import numpy as np
import matplotlib.pyplot as plt

from models import fit_least_squares, fit_ridge, mse
from data import make_synthetic, first_features


def run_synthetic(d_values, m=100, m_test=2000, D=500, noise_std=0.5, alpha=None, n_seeds=10):
    """Average train error, test error and weight norm over n_seeds datasets, for each d in d_values.

    alpha=None -> minimum-norm least squares (pinv); alpha > 0 -> ridge.
    Returns three arrays with one value per d: train_err, test_err, w_norm.
    """
    n_d = len(d_values)
    # One row per seed, one column per value of d
    train_err = np.zeros((n_seeds, n_d))
    test_err = np.zeros((n_seeds, n_d))
    w_norm = np.zeros((n_seeds, n_d))

    for s in range(n_seeds):
        X_train, y_train, X_test, y_test, w_star = make_synthetic(m, m_test, D, noise_std, seed=s)
        for j, d in enumerate(d_values):
            X_tr_d = first_features(X_train, d)
            X_te_d = first_features(X_test, d)

            # TODO 1: fit the model on (X_tr_d, y_train).
            # If alpha is None use fit_least_squares, otherwise fit_ridge with that alpha.
            # Hint: if alpha is None:
            #           w = ...
            #       else:
            #           w = ...
            w = None

            # TODO 2: store the three numbers in row s, column j.
            # Hint: train_err[s, j] = mse(X_tr_d, y_train, w); same idea for test, and np.linalg.norm(w)
            pass

    # Average over the seeds (axis=0 = down the rows): one value per d
    return train_err.mean(axis=0), test_err.mean(axis=0), w_norm.mean(axis=0)


if __name__ == "__main__":
    m, noise_std = 100, 0.5
    d_values = np.arange(1, 501)                       # d = 1, 2, ..., 500
    train, test, norm = run_synthetic(d_values, m=m, noise_std=noise_std)

    # Quick checks
    for d in [10, 50, 90, 100, 110, 200, 500]:
        j = d - 1                                      # d_values[0] is 1, so d sits at index d-1
        print(f"d={d:3d}  train={train[j]:9.4f}  test={test[j]:10.4f}  ||w||={norm[j]:8.3f}")

    # First double descent plot (a quick look; the final plots will live in plots.py)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d_values, test, label="test error (min-norm LS)")
    ax.plot(d_values, train, label="train error")
    ax.axvline(m, color="gray", linestyle="--", label="d = m")
    ax.axhline(noise_std ** 2, color="black", linestyle=":", label="Bayes risk $\\sigma^2$")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e4)          # train error after d = m is ~1e-28 (numerical zero): it drops off the bottom
    ax.set_xlabel("number of features d")
    ax.set_ylabel("mean squared error")
    ax.legend()
    fig.tight_layout()
    fig.savefig("first_double_descent.png", dpi=150)
    print("Plot saved to first_double_descent.png")
