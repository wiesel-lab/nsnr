"""Reproduces Figure 6 of the paper (Section 6.2): LW, QIS and KL-LOOCV
on synthetic Kronecker product covariances, C = kron(C1, C2) with random
positive definite factors.

Usage:
    python experiments/figure6.py

Output is saved to results/figure6.png
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from common import metrics_row, plot_panel
from src.data import generate_data
from src.estimators import loocv, lw, qis


def main():
    os.makedirs("results", exist_ok=True)
    np.random.seed(0)  # fixed seed => reproducible

    results = []
    for Mi in np.linspace(20, 100, 9, dtype=np.int32):
        print(f"M={Mi}...")
        C, X, mode = generate_data(D=10, M=Mi + 1, B=1000, mode="kron_pd")
        for est in (lw, qis, loocv):
            results.append(metrics_row(est, C, X, mode))
    df = pd.DataFrame(results)
    print(df)

    plot_panel(df, out="results/figure6.png",
               title="Kronecker product covariance")


if __name__ == "__main__":
    main()
