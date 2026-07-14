"""Covariance accuracy metrics from the paper.

All metrics depend on (C, Chat) only through the matrix ratio
Q = Chat^{-1/2} C Chat^{-1/2} (Definition 2 in the paper).

Two sets of functions are provided:
  * single-matrix versions (inputs of shape (D, D)), used by Figures 1-3;
  * batched versions (inputs of shape (B, D, D)), used by Figures 4-6.
"""

import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Single-matrix versions (D, D)
# ---------------------------------------------------------------------------

def project_pd(C, floor=1e-2):
    """Project a symmetric matrix onto the PD cone (small floor on eigenvalues)."""
    w, V = np.linalg.eigh((C + C.conj().T) / 2)
    return (V * np.maximum(w, floor)) @ V.conj().T


def ratio_eigs(C, Chat):
    """Eigenvalues of Q = Chat^{-1/2} C Chat^{-1/2} (real, positive)."""
    isq = np.linalg.inv(sqrtm(project_pd(Chat)))
    Q = isq @ C @ isq
    q = np.linalg.eigvalsh((Q + Q.conj().T) / 2)
    return np.maximum(np.real(q), 1e-12)


def d_nsnr_from_eigs(q):
    """Worst-case NSNR distance, Eq. (23), from the eigenvalues of Q."""
    qmin, qmax = q.min(), q.max()
    return 0.5 * np.log(((qmin + qmax) / 2) ** 2 / (qmin * qmax))


def d_invariant_kl_from_eigs(q):
    """Invariant (scale-free) KL, Eq. (25): (D/2) log(AM/GM) of the eigenvalues."""
    D = len(q)
    return (D / 2) * np.log(q.mean() / np.exp(np.log(q).mean()))


def d_nsnr(C, Chat):
    """Worst-case NSNR distance, Eq. (23)."""
    return d_nsnr_from_eigs(ratio_eigs(C, Chat))


def d_kl(C, Chat):
    """Gaussian KL divergence, Eq. (24)."""
    q = ratio_eigs(C, Chat)
    D = len(q)
    return 0.5 * (q.sum() - D - np.log(q).sum())


def d_invariant_kl(C, Chat):
    """Invariant (scale-free) KL divergence, Eq. (25)."""
    return d_invariant_kl_from_eigs(ratio_eigs(C, Chat))


def d_nmse(C, Chat):
    """Normalized MSE (squared Frobenius norm, scale-normalized)."""
    return np.linalg.norm(C - Chat, "fro") ** 2 / np.linalg.norm(C, "fro") ** 2


# ---------------------------------------------------------------------------
# Batched versions (B, D, D)
# ---------------------------------------------------------------------------

def proj_pd_batch(C, reg=1e-2):
    """Project a stack of symmetric matrices onto the PD cone."""
    d, u = np.linalg.eigh(C)
    d = d * (d >= 0) + reg
    return np.einsum('bij,bj,bkj->bik', u, d, u)


def kl(C_true, C_pred):
    """Gaussian KL divergence, Eq. (24), averaged over the batch."""
    C_pred = proj_pd_batch(C_pred)
    D = C_true.shape[1]
    C_inv = np.linalg.inv(C_pred)
    mult = np.matmul(C_true, C_inv)
    trace_term = np.trace(mult, axis1=1, axis2=2)
    log_det = np.linalg.slogdet(mult)[1]
    return (0.5 * (trace_term - D - log_det)).mean()


def invariant_kl(C_true, C_pred):
    """Invariant (scale-free) KL divergence, Eq. (25), averaged over the batch."""
    C_pred = proj_pd_batch(C_pred)
    D = C_true.shape[1]
    C_inv = np.linalg.inv(C_pred)
    mult = np.matmul(C_true, C_inv)
    trace_term = np.trace(mult, axis1=1, axis2=2)
    log_det = np.linalg.slogdet(mult)[1]
    return (D / 2 * np.log(trace_term / D) - 0.5 * log_det).mean()


def nsnr(C_true, C_pred):
    """Worst-case NSNR distance, Eq. (23), averaged over the batch."""
    try:
        C_pred = proj_pd_batch(C_pred)
        C_si = np.linalg.cholesky(np.linalg.inv(C_pred))
        Q = np.einsum('bji,bjk,bkr->bir', C_si, C_true, C_si)
        eigs = np.linalg.eigh(Q)[0]
        kappa = np.amax(eigs, 1) / np.amin(eigs, 1)
        return (-0.5 * np.log(4 * kappa / (kappa + 1) ** 2)).mean()
    except np.linalg.LinAlgError:
        return np.nan


def nmse(C_true, C_pred):
    """Normalized MSE, averaged over the batch."""
    d = ((C_true - C_pred) ** 2).mean((1, 2)) / ((C_true) ** 2).mean((1, 2))
    return d.mean()


def compute_auc(C_pred, y_test, num_experiments=20000):
    """Negative AUC of the adaptive matched filter over random targets."""
    B, D, _ = C_pred.shape
    C_pred = proj_pd_batch(C_pred)
    inv_cov = np.linalg.inv(C_pred)
    amp = 0.6
    auc_scores = []
    for _ in range(num_experiments):
        target = np.random.rand(D)
        y = np.zeros(B)
        y[:B // 2] = 1
        np.random.shuffle(y)
        ampy = amp * y
        y_test = y_test.reshape(B, D)
        y_test_with_target = y_test + ampy[:, None] * target[None, :]
        score = ((target.reshape(-1, 1, D) @ inv_cov @ y_test_with_target.reshape(B, D, 1)) ** 2 /
                 (target.reshape(-1, 1, D) @ inv_cov @ target.reshape(-1, D, 1)))
        auc_scores.append(roc_auc_score(y, score.reshape(B)))
    return -1 * np.mean(auc_scores)
