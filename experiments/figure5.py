"""Reproduces Figure 5 of the paper (Section 6.2): LW, QIS and KL-LOOCV
on real covariance matrices from the Pavia University hyperspectral
dataset (Trees label).

Requires the dataset; download it first:
    bash data/download_data.sh

Usage:
    python experiments/figure5.py

Output is saved to results/figure5.png
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from common import metrics_row, plot_panel
from src.data import generate_pavia_data
from src.estimators import loocv, lw, qis


def main(background_class=4):
    os.makedirs("results", exist_ok=True)
    np.random.seed(0)  # fixed seed => reproducible

    results = []
    for Mi in np.linspace(20, 100, 9, dtype=np.int32):
        print(f"M={Mi}...")
        C, X, mode = generate_pavia_data(M=Mi + 1, B=300,
                                         background_class=background_class,
                                         downsampling_factor=3)
        for est in (lw, qis, loocv):
            results.append(metrics_row(est, C, X, mode))
    df = pd.DataFrame(results)
    print(df)

    plot_panel(df, out="results/figure5.png",
               title="Pavia University hyperspectral, Trees label",
               ylim_kl=[16, 104, 0, 38])


if __name__ == "__main__":
    main()
