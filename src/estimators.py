"""Covariance estimators compared in the paper (Section 5).

All estimators take samples X of shape (B, M, D) -- B independent
realizations of M samples in dimension D -- and return a (B, D, D)
stack of covariance estimates.
"""

import math

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS


def scm(X, reg=1e-5):
    """Regularized sample covariance matrix."""
    B, M, D = X.shape
    S = np.swapaxes(X, 1, 2) @ X / M
    return (1 - reg) * S + reg * np.eye(D)[None, :, :]


def lw(X):
    """Ledoit-Wolf shrinkage (norm-based selection of the shrinkage parameter)."""
    B, M, D = X.shape
    Chat = np.zeros((B, D, D))
    for b in range(B):
        Chat[b] = LedoitWolf(assume_centered=True).fit(X[b, :, :]).covariance_
    return Chat


def oas(X):
    """Oracle Approximating Shrinkage."""
    B, M, D = X.shape
    Chat = np.zeros((B, D, D))
    for b in range(B):
        Chat[b] = OAS(assume_centered=True).fit(X[b, :, :]).covariance_
    return Chat


def _qis_single(Y, k=None):
    """Quadratic-Inverse Shrinkage (QIS) estimator of Ledoit and Wolf.

    Y is a pd.DataFrame with N rows (samples) and p columns (variables).
    """
    N = Y.shape[0]
    p = Y.shape[1]

    k = 0

    n = N - k  # adjust effective sample size
    c = p / n  # concentration ratio

    sample = pd.DataFrame(np.matmul(Y.T.to_numpy(), Y.to_numpy()) / n)
    sample = (sample + sample.T) / 2  # make symmetrical

    # Spectral decomposition
    lambda1, u = np.linalg.eigh(sample)
    lambda1 = lambda1.real.clip(min=0)
    dfu = pd.DataFrame(u, columns=lambda1)
    dfu.sort_index(axis=1, inplace=True)
    lambda1 = dfu.columns

    # Quadratic-Inverse Shrinkage estimator of the covariance matrix
    h = (min(c ** 2, 1 / c ** 2) ** 0.35) / p ** 0.35  # smoothing parameter
    invlambda = 1 / lambda1[max(1, p - n + 1) - 1:p]  # inverse of (non-null) eigenvalues
    dfl = pd.DataFrame()
    dfl['lambda'] = invlambda
    Lj = dfl[np.repeat(dfl.columns.values, min(p, n))]
    Lj = pd.DataFrame(Lj.to_numpy())
    Lj_i = Lj.subtract(Lj.T)  # like (1/lambda_j)-(1/lambda_i)

    theta = Lj.multiply(Lj_i).div(Lj_i.multiply(Lj_i).add(
        Lj.multiply(Lj) * h ** 2)).mean(axis=0)  # smoothed Stein shrinker
    Htheta = Lj.multiply(Lj * h).div(Lj_i.multiply(Lj_i).add(
        Lj.multiply(Lj) * h ** 2)).mean(axis=0)  # its conjugate
    Atheta2 = theta ** 2 + Htheta ** 2

    if p <= n:
        delta = 1 / ((1 - c) ** 2 * invlambda + 2 * c * (1 - c) * invlambda * theta
                     + c ** 2 * invlambda * Atheta2)  # optimally shrunk eigenvalues
        delta = delta.to_numpy()
    else:
        delta0 = 1 / ((c - 1) * np.mean(invlambda.to_numpy()))  # shrinkage of null eigenvalues
        delta = np.repeat(delta0, p - n)
        delta = np.concatenate((delta, 1 / (invlambda * Atheta2)), axis=None)

    deltaQIS = delta * (sum(lambda1) / sum(delta))  # preserve trace

    sigmashat = np.matmul(np.matmul(dfu.to_numpy(), np.diag(deltaQIS)),
                          dfu.T.to_numpy().conjugate())
    return sigmashat


def qis(X):
    """Quadratic-Inverse Shrinkage applied per realization."""
    B, M, D = X.shape
    Chat = np.zeros((B, D, D))
    for b in range(B):
        Chat[b] = _qis_single(pd.DataFrame(X[b, :, :]), k=0)
    return Chat


def _loocv_loglike(X, alpha):
    """Leave-one-out KL objective J(alpha), Eqs. (35)-(37).

    Uses the Sherman-Morrison rank-one update so the full leave-one-out
    objective costs O(D^3 + M D^2) per alpha instead of one inversion
    per left-out sample.
    """
    D, N = X.shape
    S = X @ X.T / N
    T = np.trace(S) / D * np.eye(D)
    C = (1 - alpha) * S + alpha * T
    invC = np.linalg.inv(C)
    z = np.einsum('ji,jk,ki->i', X, invC, X)  # z_i = x_i^T C^-1 x_i
    trace_term = np.mean(z / (1 - (1 - alpha) / N * z))
    log_det = np.log(np.linalg.det(C))
    return trace_term + log_det


def loocv(X):
    """KL-LOOCV shrinkage covariance estimator (Algorithm 1).

    Same shrinkage family as Ledoit-Wolf, Chat = (1-alpha) S + alpha tr(S)/D I,
    but alpha is selected by minimizing a leave-one-out approximation of the
    KL divergence instead of a norm-based criterion.
    """
    alphas = np.logspace(-3, -.01, 20)
    B, M, D = X.shape
    Chat = np.zeros((B, D, D))
    for b in range(B):
        Xi = X[b, :, :].T
        distances = np.array([_loocv_loglike(Xi, alpha) for alpha in alphas])
        optimal_alpha = alphas[np.argmin(distances)]
        S = Xi @ Xi.T / M
        Chat[b] = (1 - optimal_alpha) * S + optimal_alpha * np.trace(S) / D * np.eye(D)
    return Chat
