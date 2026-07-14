"""Reproduces Figure 2 of the paper (Section 6.1, first experiment):
scatter plots of the worst-case NSNR distance against the KL and
normalized Frobenius (MSE) metrics over 1000 sample covariances.

Usage:
    python experiments/figure2.py

Output is saved to results/figure2.png
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

from src.data import sample_cov
from src.metrics import d_kl, d_nmse, d_nsnr, project_pd


def main(D=10, M=200, n_trials=1000):
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(0)  # fixed seed => reproducible

    O = np.ones((D, D))
    C = 0.5 * O + np.eye(D)  # single very large eigenvalue
    nsnr_v, kl_v, mse_v = [], [], []
    for _ in range(n_trials):
        Chat = project_pd(sample_cov(C, M, rng))
        nsnr_v.append(d_nsnr(C, Chat))
        kl_v.append(d_kl(C, Chat))
        mse_v.append(d_nmse(C, Chat))
    nsnr_v, kl_v, mse_v = map(np.array, (nsnr_v, kl_v, mse_v))

    r_kl = np.corrcoef(kl_v, nsnr_v)[0, 1]
    r_mse = np.corrcoef(mse_v, nsnr_v)[0, 1]
    print(f"Pearson KL-NSNR = {r_kl:.2f}, MSE-NSNR = {r_mse:.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].scatter(kl_v, nsnr_v, s=10, alpha=0.5)
    ax[0].set_xlabel("KL")
    ax[0].set_ylabel("NSNR")
    ax[0].set_title(f"KL vs NSNR (r = {r_kl:.2f})")
    ax[1].scatter(mse_v, nsnr_v, s=10, alpha=0.5, color="tab:orange")
    ax[1].set_xlabel("MSE")
    ax[1].set_ylabel("NSNR")
    ax[1].set_title(f"MSE vs NSNR (r = {r_mse:.2f})")
    fig.tight_layout()
    fig.savefig("results/figure2.png", dpi=300, bbox_inches="tight")
    print("Saved results/figure2.png")


if __name__ == "__main__":
    main()
