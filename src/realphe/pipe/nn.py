import numpy as np


def focal_loss_bias_init(
    Y_train: np.ndarray,
    alpha: float = 0.5,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 500
) -> np.ndarray:
    """
    Vectorized computation of output biases for sigmoid output layer
    under focal loss initialization, with optional label smoothing.

    Parameters
    ----------
    Y_train : ndarray of shape (N, C)
        Binary target matrix (samples × labels).
    alpha : float in (0,1)
        Class weight parameter in focal loss.
    gamma : float >= 0
        Focusing parameter in focal loss.
    label_smoothing : float in [0, 0.5)
        Label smoothing parameter ε.
        - ε=0 means no smoothing (default).
        - Targets are adjusted as:
          p_smooth = p*(1-2ε) + ε
    tol : float
        Convergence tolerance for numerical root finding.
    max_iter : int
        Maximum iterations for Newton-Raphson.

    Returns
    -------
    bias : ndarray of shape (C,)
        Initial biases for sigmoid output layer.
    """
    N, C = Y_train.shape
    # empirical prevalence per class
    p_hat = Y_train.mean(axis=0)

    # apply label smoothing: p -> p*(1-2ε)+ε
    if label_smoothing > 0:
        p_hat = p_hat * (1 - 2 * label_smoothing) + label_smoothing

    # Special case: gamma=0 => closed form vectorized
    if gamma == 0.0:
        num = alpha * p_hat
        den = alpha * p_hat + (1 - alpha) * (1 - p_hat)
        y_star = num / den
        y_star = np.clip(y_star, 1e-6, 1 - 1e-6)
        return np.log(y_star / (1 - y_star))

    # For gamma > 0: Newton iterations in parallel
    num = alpha * p_hat
    den = alpha * p_hat + (1 - alpha) * (1 - p_hat)
    y = num / den
    y = np.clip(y, 1e-6, 1 - 1e-6)

    def f_eval(z):
        term1 = (1 - p_hat) * (1 - alpha) * (z**gamma) / (1 - z)
        term2 = -p_hat * alpha * ((1 - z)**gamma) / z
        term3 = gamma * (
            p_hat * alpha * (1 - z)**(gamma - 1) * np.log(z)
            - (1 - p_hat) * (1 - alpha) * (z**(gamma - 1)) * np.log(1 - z)
        )
        return term1 + term2 + term3

    for _ in range(max_iter):
        f = f_eval(y)
        eps = 1e-6
        fprime = (f_eval(np.clip(y + eps, 1e-6, 1 - 1e-6)) -
                  f_eval(np.clip(y - eps, 1e-6, 1 - 1e-6))) / (2 * eps)

        step = f / (fprime + 1e-12)
        y_new = np.clip(y - step, 1e-6, 1 - 1e-6)

        if np.all(np.abs(y_new - y) < tol):
            y = y_new
            break
        y = y_new

    return np.log(y / (1 - y))
