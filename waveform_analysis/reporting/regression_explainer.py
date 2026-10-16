"""Population reports for a global regression surrogate; no single-event plots."""
from __future__ import annotations

import numpy as np


def plot_explanation(directory, tree, names, original, surrogate, metrics, shapelets):
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(18, max(5, 2 * tree.get_depth())))
    plot_tree(tree, feature_names=names, filled=True, rounded=True, precision=3, ax=ax, fontsize=7)
    ax.set_title('Global shapelet surrogate — leaf value: predicted correction [ps]')
    fig.savefig(directory / 'tree.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hexbin(original, surrogate, gridsize=40, mincnt=1, cmap='viridis')
    low, high = min(np.min(original), np.min(surrogate)), max(np.max(original), np.max(surrogate))
    axes[0].plot([low, high], [low, high], 'k--', linewidth=1)
    axes[0].set(xlabel='Original correction [ps]', ylabel='Surrogate correction [ps]', title='Blind fidelity')
    for ax, key, title in zip(axes[1:], ['ctr_ps', 'rmse_ps'], ['Blind CTR [ps]', 'Blind RMSE [ps]']):
        ax.bar([m['model'] for m in metrics], [m[key] for m in metrics], color=['gray', '#0072B2', '#E69F00'])
        ax.set(ylabel=title, title='Same held-out events')
    fig.tight_layout()
    fig.savefig(directory / 'performance.png', dpi=160)
    plt.close(fig)

    used = np.unique(tree.tree_.feature[tree.tree_.feature >= 0])
    if not len(used):
        return  # A constant tree genuinely has no explanatory subsequences.
    n = len(shapelets['shapelet_id'])
    fig, axes = plt.subplots(len(used), 1, figsize=(9, 2.2 * len(used)), squeeze=False)
    for ax, feature in zip(axes[:, 0], used):
        s = feature % n
        length, dilation = int(shapelets['lengths'][s]), int(shapelets['dilations'][s])
        ax.plot(np.arange(length) * dilation, shapelets['values'][s, 0, :length], marker='.')
        ax.set(title=f"{names[feature]} | z-normalized: {bool(shapelets['normalizes'][s])}",
               xlabel='Offset on retained sample grid', ylabel='Shapelet amplitude')
    fig.tight_layout()
    fig.savefig(directory / 'used_shapelets.png', dpi=140)
    plt.close(fig)
