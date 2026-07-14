"""Reproduces Figure 3 of the paper (Section 6.1, second experiment):
each metric against the empirical probability of detection P_d of the
adaptive matched filter (AMF) detector.

Usage:
    python experiments/figure3.py

Output is saved to results/figure3.png
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz

from src.data import sample_cov
from src.metrics import d_kl, d_nmse, d_nsnr, project_pd


def amf_pd(C, Chat, rng, snr_db=10.0, pfa=1e-3, n_targets=5, n_mc=2000):
    """Empirical probability of detection of the AMF detector that uses Chat,
    averaged over several random targets, at the given Pfa and SNR."""
    D = C.shape[0]
    Cinv_hat = np.linalg.inv(project_pd(Chat))
    L = np.linalg.cholesky(C)
    snr = 10 ** (snr_db / 10)
    pds = []
    for _ in range(n_targets):
        s = rng.standard_normal(D)
        s = s / np.sqrt(s @ np.linalg.inv(C) @ s)  # normalize output SNR
        a = np.sqrt(snr)
        w = Cinv_hat @ s
        # H0 noise samples -> threshold for the target Pfa
        n0 = L @ rng.standard_normal((D, n_mc))
        t0 = np.abs(w @ n0) ** 2 / (w @ C @ w)
        thr = np.quantile(t0, 1 - pfa)
        # H1 samples (target present)
        n1 = L @ rng.standard_normal((D, n_mc))
        y1 = a * s[:, None] + n1
        t1 = np.abs(w @ y1) ** 2 / (w @ C @ w)
        pds.append(np.mean(t1 > thr))
    return float(np.mean(pds))


def main(D=10, rho=0.9, snr_db=10.0, pfa=1e-3, B=900):
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(0)  # fixed seed => reproducible

    C = toeplitz(rho ** np.arange(D))
    nsnr_v, kl_v, nmse_v, pd_v = [], [], [], []
    for _ in range(B):
        M = int(rng.integers(18, 36))  # 18..35 samples
        Chat = project_pd(sample_cov(C, M, rng))
        nsnr_v.append(d_nsnr(C, Chat))
        kl_v.append(d_kl(C, Chat))
        nmse_v.append(d_nmse(C, Chat))
        pd_v.append(amf_pd(C, Chat, rng, snr_db=snr_db, pfa=pfa))
    nsnr_v, kl_v, nmse_v, pd_v = map(np.array, (nsnr_v, kl_v, nmse_v, pd_v))

    r_nsnr = np.corrcoef(nsnr_v, pd_v)[0, 1]
    r_kl = np.corrcoef(kl_v, pd_v)[0, 1]
    r_nmse = np.corrcoef(nmse_v, pd_v)[0, 1]
    print(f"Pearson with Pd: NSNR={r_nsnr:.2f}, KL={r_kl:.2f}, NMSE={r_nmse:.2f}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, x, name, r, col in [
        (ax[0], nsnr_v, "NSNR", r_nsnr, "tab:blue"),
        (ax[1], kl_v, "KL", r_kl, "tab:green"),
        (ax[2], nmse_v, "NMSE", r_nmse, "tab:red"),
    ]:
        a.scatter(x, pd_v, s=10, alpha=0.5, color=col)
        a.set_xlabel(name)
        a.set_ylabel(r"$P_d$")
        a.set_title(f"{name} vs $P_d$ (r = {r:.2f})")
    fig.tight_layout()
    fig.savefig("results/figure3.png", dpi=300, bbox_inches="tight")
    print("Saved results/figure3.png")


if __name__ == "__main__":
    main()
