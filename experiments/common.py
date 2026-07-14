"""Shared helpers for the estimator-comparison panels (Figures 4-6)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np

from src.metrics import compute_auc, kl, nmse, nsnr


def metrics_row(est, C, X, mode):
    """One results-table row: all four metrics for estimator `est`.

    The last sample of each realization is held out as the test vector
    for the detection AUC.
    """
    X, y = X[:, :-1, :], X[:, -1, :]
    B, M, D = X.shape
    Chat = est(X)
    return {'EST': est.__name__,
            'M': M,
            'B': B,
            'D': D,
            'mode': mode,
            'NMSE': np.round(nmse(C, Chat), 2),
            'KL': np.round(kl(C, Chat), 2),
            'NSNR': np.round(nsnr(C, Chat), 2),
            'Neg-AUC': np.round(compute_auc(Chat, y), 2)}


# Per-estimator style: distinct color, linestyle and marker so overlapping
# curves stay distinguishable.
ESTIMATOR_STYLE = {
    'lw':    dict(color='tab:blue',   linestyle='-',  marker='o', markersize=5),
    'qis':   dict(color='tab:orange', linestyle='--', marker='s', markersize=5),
    'loocv': dict(color='tab:green',  linestyle='-.', marker='^', markersize=5),
}

LEGEND_NAMES = {'lw': 'LW', 'qis': 'QIS', 'loocv': 'LOOCV'}


def plot_panel(df, out, title=None, ylim_kl=None):
    """1x4 panel of NSNR / KL / Neg-AUC / NMSE versus M for LW, QIS and LOOCV."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    fig.supxlabel('M')

    metrics_lst = ['NSNR', 'KL', 'Neg-AUC', 'NMSE']
    Mlist = df.loc[df['EST'] == 'loocv', 'M'].to_numpy()
    estimators = ['lw', 'qis', 'loocv']

    for i, metric in enumerate(metrics_lst):
        for est in estimators:
            y = df.loc[df['EST'] == est, metric].to_numpy()
            axes[i].plot(Mlist, y, label=LEGEND_NAMES[est],
                         linewidth=1.8, **ESTIMATOR_STYLE[est])
        axes[i].legend(loc='best', fontsize=9)
        axes[i].set_title(metric)
        axes[i].grid(alpha=0.25)

    if ylim_kl is not None:
        axes[1].axis(ylim_kl)

    if title is not None:
        fig.suptitle(title, y=1.02)

    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')
