"""Reproduces Figure 1 of the paper: tightness of the bound
d_NSNR <= d_invariant_KL (Theorem 2), as the interior eigenvalues of Q
sweep between q_min and q_max.

Usage:
    python experiments/figure1.py

Output is saved to results/figure1.png
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import d_invariant_kl_from_eigs, d_nsnr_from_eigs


def main(D=10, qmin=1.0, qmax=10.0, n=200):
    os.makedirs("results", exist_ok=True)

    positions = np.linspace(qmin, qmax, n)
    ratios = []
    for p in positions:
        q = np.concatenate(([qmin], np.full(D - 2, p), [qmax]))  # interior all at p
        dn = d_nsnr_from_eigs(q)
        di = d_invariant_kl_from_eigs(q)
        ratios.append(dn / di if di > 1e-12 else np.nan)
    ratios = np.array(ratios)
    am = (qmin + qmax) / 2

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(positions, ratios, lw=2)
    ax.axvline(am, ls="--", color="gray", label=f"arithmetic mean = {am:g}")
    ax.set_xlabel("interior eigenvalue value")
    ax.set_ylabel(r"$d_{\rm NSNR} / d_{\rm invariant\ KL}$")
    ax.set_title("Tightness of the bound in Theorem 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figure1.png", dpi=300, bbox_inches="tight")
    print(f"Saved results/figure1.png (max ratio ~ {np.nanmax(ratios):.3f} at the mean)")


if __name__ == "__main__":
    main()
