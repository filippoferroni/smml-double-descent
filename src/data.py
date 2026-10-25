"""Datasets for the double descent experiments (numpy only).

Synthetic data: noisy linear labels, "weak features" setup.
    x ~ N(0, I_D),  y = w_star^T x + noise,  noise ~ N(0, noise_std^2)
The model with complexity d only sees the first d of the D features.
The Bayes risk (best possible test error) is noise_std^2 (slides "4. statRisk", p. 1-2).
"""
import numpy as np


def make_synthetic(m, m_test, D, noise_std, seed):
    """Generate a training set (m examples) and an independent test set (m_test examples).

    Returns X_train (m, D), y_train (m,), X_test (m_test, D), y_test (m_test,), w_star (D,).
    """
    rng = np.random.default_rng(seed)        # random generator with a fixed seed: reproducible

    # True weight vector, normalized to norm 1 so that the signal has unit variance
    w_star = rng.normal(size=D)
    w_star = w_star / np.linalg.norm(w_star)

    # TODO 1: draw X_train with m rows and D columns from the standard normal distribution.
    # Hint: same function used for w_star, with size=(m, D).
    X_train = rng.normal(size=(m, D))

    # TODO 2: labels y_train = X_train w_star + noise, with noise of standard deviation noise_std.
    # Hint: noise_std * rng.normal(size=m) gives m noise values.
    y_train = X_train @ w_star + noise_std * rng.normal(size=m)

    # Test set: same distribution, drawn independently (no example is shared with training)
    X_test = rng.normal(size=(m_test, D))
    y_test = X_test @ w_star + noise_std * rng.normal(size=m_test)

    return X_train, y_train, X_test, y_test, w_star


def first_features(X, d):
    """Keep only the first d columns (features) of X."""
    # TODO 3: return the first d columns of X.
    # Hint: slicing X[:, :d] means "all rows, columns from 0 to d-1".
    return X[:, :d]


if __name__ == "__main__":
    # Self-checks. Run: python src/data.py
    X_tr, y_tr, X_te, y_te, w_star = make_synthetic(m=100, m_test=2000, D=500, noise_std=0.5, seed=0)
    print("Check A, shapes (expected (100, 500) (100,) (2000, 500) (2000,)):",
          X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)

    print("Check B, shape with d=30 (expected (100, 30)):", first_features(X_tr, 30).shape)

    # Check C: the true w_star has test error close to the Bayes risk noise_std^2 = 0.25
    test_error_true = np.mean((X_te @ w_star - y_te) ** 2)
    print("Check C, test error of w_star (expected about 0.25):", test_error_true)

    # Check D: same seed, same data (reproducibility)
    X_tr2 = make_synthetic(m=100, m_test=2000, D=500, noise_std=0.5, seed=0)[0]
    print("Check D, same seed gives same data (expected True):", np.array_equal(X_tr, X_tr2))
