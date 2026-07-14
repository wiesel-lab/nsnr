"""Ground-truth covariance models and sample generation (Section 6)."""

import os

import numpy as np
from scipy.io import loadmat
from scipy.linalg import toeplitz


def toeplitz_from_first_row(X):
    B, D = X.shape
    i = np.arange(D).reshape(-1, 1)
    j = np.arange(D).reshape(1, -1)
    idx = np.abs(j - i)
    return X[:, idx]


def generate_data(D, M, B, mode):
    """B ground-truth covariances of size DxD and M Gaussian samples of each.

    Returns (C, X, mode) with C of shape (B, D, D) and X of shape (B, M, D).
    """
    if mode == 'toeplitz':
        first_row = np.random.randn(B, D)
        C = toeplitz_from_first_row(first_row)
        eigvals, eigvecs = np.linalg.eigh(C)
        min_eigval = np.amin(eigvals, -1)
        C = C + (0.1 - min_eigval)[:, None, None] * np.eye(D)[None, :, :]
    elif mode == 'low rank':
        R = 5
        C = 10 * np.random.randn(B, D, R)
        C = (C @ np.swapaxes(C, 1, 2)) / R + np.random.rand(B)[:, None, None] * np.eye(D)[None, :, :]
    elif mode == 'ar1':
        # AR(1) process: C[i,j] = rho^|i-j| with a fixed rho. One ground truth
        # shared across all B realizations; only the sample sets differ.
        rho = 0.9
        row = rho ** np.arange(D)
        C_single = toeplitz(row)
        C = np.tile(C_single[None, :, :], (B, 1, 1))
    elif mode == 'kron_pd':
        # Kronecker product of two random positive definite matrices,
        # C = kron(C1, C2), with C1 of size 2x2 and C2 of size D/2 x D/2.
        # Both factors are drawn freshly for each of the B trials.
        D1 = 2
        D2 = D // D1
        C = np.zeros((B, D, D))
        for b in range(B):
            a = np.random.randn(D1, D1)
            C1 = a @ a.T / D1 + 1e-1 * np.eye(D1)
            a = np.random.randn(D2, D2)
            C2 = a @ a.T / D2 + 1e-1 * np.eye(D2)
            C[b] = np.kron(C1, C2)
    else:
        raise ValueError(f'unknown mode: {mode}')
    sC = np.linalg.cholesky(C)
    W = np.random.randn(B, D, M)
    X = sC @ W
    X = np.swapaxes(X, 1, 2)
    return C, X, mode


def generate_pavia_data(M=80, B=100, background_class=4, downsampling_factor=4,
                        data_dir='data'):
    """Background covariance and samples from the Pavia University dataset.

    Requires PaviaU.mat and PaviaU_gt.mat in `data_dir`; run
    `bash data/download_data.sh` to obtain them.
    """
    try:
        Xall = loadmat(os.path.join(data_dir, 'PaviaU.mat'))['paviaU']
        yall = loadmat(os.path.join(data_dir, 'PaviaU_gt.mat'))['paviaU_gt']
    except FileNotFoundError as e:
        raise FileNotFoundError(
            'Pavia University dataset not found. Run: bash data/download_data.sh'
        ) from e
    Xall = Xall[:, :, ::downsampling_factor]
    Xv = Xall.reshape(-1, Xall.shape[2]) / 100
    yv = yall.reshape(-1)
    D = Xall.shape[2]
    Xb = Xv[yv == background_class, :]
    N = Xb.shape[0] // 2
    # ground truth from the first half of the pixels
    Xbwm = Xb[:N] - np.mean(Xb[:N], axis=0)
    C = Xbwm.T @ Xbwm / Xbwm.shape[0]
    C = np.tile(C[None, :, :], (B, 1, 1))
    # training samples drawn from the second half
    Xtrain = Xb[N:]
    X = np.zeros((B, M, D))
    for i in range(B):
        X[i] = Xtrain[np.random.randint(low=0, high=Xtrain.shape[0], size=(M,)), :]
        X[i] = X[i] - np.mean(X[i], axis=0)
    return C, X, 'pavia' + str(background_class)


def sample_cov(C, M, rng):
    """Sample covariance of M Gaussian samples drawn from N(0, C)."""
    D = C.shape[0]
    X = rng.multivariate_normal(np.zeros(D), C, size=M)
    return np.cov(X, rowvar=False)
