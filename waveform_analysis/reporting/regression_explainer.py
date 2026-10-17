"""Population reports for a global regression surrogate; no single-event plots."""
from __future__ import annotations

import numpy as np


def plot_explanation(directory, tree, names, original, surrogate, metrics, shapelets, contributions, formulation):
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(18, max(5, 2 * tree.get_depth())))
    plot_tree(tree, feature_names=names, filled=True, rounded=True, precision=3, ax=ax, fontsize=7)
    ax.set_title('Approximate tree on ORIGINAL frozen features — leaf value: correction [ps]')
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

    if contributions is not None:
        order = contributions['ranking'][:20]
        labels = [f"Shapelet {i}" if shapelets is not None else f"Feature {i}" for i in order]
        fig, axes = plt.subplots(1, 2, figsize=(13, max(4, .3 * len(order))))
        axes[0].barh(labels, contributions['mean_absolute_contribution_ps'][order], color='#0072B2')
        axes[0].set(xlabel='Mean absolute net contribution [ps]', title='Original model: global reliance')
        axes[0].invert_yaxis()
        axes[1].barh(labels, contributions['mean_contribution_ps'][order],
                     xerr=contributions['std_contribution_ps'][order], color='#009E73', alpha=.8)
        axes[1].axvline(0, color='gray', linewidth=1)
        axes[1].set(xlabel='Mean contribution ± population SD [ps]', title='Original model: signed contributions')
        axes[1].invert_yaxis()
        fig.tight_layout()
        fig.savefig(directory / 'contributions.png', dpi=160)
        plt.close(fig)
    if shapelets is None:
        return
    # Original RDST feature order is [distance, location, occurrence] per shapelet.
    used = np.unique(tree.tree_.feature[tree.tree_.feature >= 0] // 3)
    if contributions is not None:
        used = np.unique(np.r_[used, contributions['ranking'][:10]])
    if not len(used):
        return
    fig, axes = plt.subplots(len(used), 1, figsize=(9, 2.2 * len(used)), squeeze=False)
    for ax, s in zip(axes[:, 0], used):
        length, dilation = int(shapelets['lengths'][s]), int(shapelets['dilations'][s])
        for channel, values in enumerate(shapelets['values'][s]):
            ax.plot(np.arange(length) * dilation, values[:length], marker='.', label=f'component {channel + 1}')
        ax.set(title=f"Original shapelet {s} | {formulation} | z-normalized: {bool(shapelets['normalizes'][s])}",
               xlabel='Offset on original retained sample grid', ylabel='Saved shapelet amplitude')
        if len(shapelets['values'][s]) > 1:
            ax.legend()
    fig.tight_layout()
    fig.savefig(directory / 'used_shapelets.png', dpi=140)
    plt.close(fig)
