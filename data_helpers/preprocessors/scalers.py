import numpy as np

def normalize_feats(X: np.ndarray, min_quantile: float, max_quantile: float):
    X_median = np.median(X, axis=1, keepdims=True)
    X_iqr = np.quantile(X, q=max_quantile, axis=1, keepdims=True) - np.quantile(X, q=min_quantile, axis=1,
                                                                                keepdims=True)
    X_iqr[X_iqr == 0] = 1
    X_norm = (X - X_median) / X_iqr
    return X_norm, X_median, X_iqr


def normalize_other(y: np.ndarray, X_median: np.ndarray, X_iqr: np.ndarray):
    norm_y = (y - X_median) / X_iqr
    return norm_y


def denormalize_feats(X: np.ndarray, X_median: np.ndarray, X_iqr: np.ndarray):
    y = X * X_iqr + X_median
    return y