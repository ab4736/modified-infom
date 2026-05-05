"""
Extra visualizations to strengthen paper story.
Generates:
  fig5_pretraining_curves.pdf  -- pretraining loss: MLP vs Att-w4
  fig6_tsne_by_phase.pdf       -- t-SNE recolored: discrete phase clusters
  fig7_final_bar_2seed.pdf     -- clean 2-seed bar chart (headline result)
Run: python3 make_extra_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = 'paper'
os.makedirs(OUT, exist_ok=True)

WORK = 'exp'
COLORS = {
    'MLP':         '#1f77b4',
    'Att-w4':      '#2ca02c',
    'OZ-MLP':      '#17becf',
    'OZ-Att-w4':   '#d62728',
    'OZ-Att-w8':   '#9467bd',
    'MoG':         '#ff7f0e',
}

# ── Fig 5: Pretraining loss curves ────────────────────────────────────────────
def fig_pretraining():
    runs = {
        'MLP':     'scene3_mlp_seed0/sd000_s_7633776.0.20260503_113221/pretraining_train.csv',
        'Att-w4':  'scene3_attention_seed0/sd000_s_7633778.0.20260503_120218/pretraining_train.csv',
        'OZ-Att-w4': 'scene3_onlinez_att_w4_v3/scene3_onlinez_att_w4_v3/sd000_s_7670792.0.20260504_102734/pretraining_train.csv',
    }
    col = 'training/flow_occupancy/flow_matching_loss'
    kl_col = 'training/flow_occupancy/kl_loss'

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    for label, relpath in runs.items():
        path = os.path.join(WORK, relpath)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        x = df['step'] / 1e6
        # smooth
        w = 5
        flow_loss = df[col].rolling(w, min_periods=1).mean()
        kl_loss   = df[kl_col].rolling(w, min_periods=1).mean()
        axes[0].plot(x, flow_loss, label=label, color=COLORS[label], linewidth=1.8)
        axes[1].plot(x, kl_loss,  label=label, color=COLORS[label], linewidth=1.8)

    for ax, title, ylabel in zip(axes,
        ['Flow matching loss', 'KL loss (intention encoder)'],
        ['Loss', 'KL divergence']):
        ax.set_xlabel('Pretraining steps (M)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=9)

    fig.suptitle('Pretraining dynamics — Scene3', fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT, 'fig5_pretraining_curves.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    fig.savefig(out.replace('.pdf','.png'), bbox_inches='tight', dpi=150)
    print(f'Saved {out}')
    plt.close(fig)


# ── Fig 6: t-SNE recolored by discrete phase ─────────────────────────────────
def fig_tsne_phase():
    """Reload saved z arrays from a pickle if available, else skip."""
    import glob, pickle

    # We'll re-generate z embeddings quickly from saved numpy data if available.
    # Since we don't cache z, we'll load the raw t-SNE png and annotate it with
    # a cleaner caption note — or we re-run a small version.
    # For now, create a version that shows MLP vs Att with DISCRETE phase bins
    # (quartiles of trajectory position) instead of continuous color.

    # Try to find any saved z data
    cache = 'paper/tsne_cache.npz'
    if not os.path.exists(cache):
        print('  t-SNE cache not found, skipping fig6 (run tsne_viz.py first)')
        return

    data = np.load(cache)
    # expected keys: mlp_emb, mlp_phase, att_emb, att_phase
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    cmap = plt.cm.Set1

    for ax, key_emb, key_phase, title in [
        (axes[0], 'mlp_emb', 'mlp_phase', 'MLP encoder'),
        (axes[1], 'att_emb', 'att_phase', 'Attention (w=4)'),
    ]:
        emb = data[key_emb]
        phase = data[key_phase]
        # Bin into 4 discrete phases
        bins = np.percentile(phase, [25, 50, 75])
        phase_bin = np.digitize(phase, bins)
        phase_labels = ['Early', 'Mid-early', 'Mid-late', 'Late']
        for b in range(4):
            mask = phase_bin == b
            ax.scatter(emb[mask,0], emb[mask,1],
                      c=[cmap(b/3)], s=8, alpha=0.7, label=phase_labels[b])
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('t-SNE dim 0', fontsize=9)
        ax.set_ylabel('t-SNE dim 1', fontsize=9)
        ax.legend(fontsize=7, markerscale=2)
        ax.tick_params(labelsize=8)

    fig.suptitle('Latent intentions colored by trajectory phase (scene3)', fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT, 'fig6_tsne_phase.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    fig.savefig(out.replace('.pdf','.png'), bbox_inches='tight', dpi=150)
    print(f'Saved {out}')
    plt.close(fig)


# ── Fig 7: Clean 2-seed headline bar chart ───────────────────────────────────
def fig_headline_bar():
    """2-seed means for scene3, showing the key comparison cleanly."""

    # scene3 results, 2 seeds (s0 and s1, both fully complete)
    results = {
        'MLP\n(baseline)':    [0.60, 0.84],
        'MoG-4\n(offline)':   [0.70, 0.68],
        'Att-w4\n(offline)':  [0.68, 0.56],
        'OZ-MLP\n(online)':   [0.46, 0.74],
        'OZ-Att-w4\n(online)':[0.88, 0.78],
        'OZ-Att-w8\n(online)':[0.84, 0.80],
    }
    bar_colors = [
        COLORS['MLP'], COLORS['MoG'], COLORS['Att-w4'],
        COLORS['OZ-MLP'], COLORS['OZ-Att-w4'], COLORS['OZ-Att-w8'],
    ]

    labels = list(results.keys())
    means  = [np.mean(v) for v in results.values()]
    stds   = [np.std(v)  for v in results.values()]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=bar_colors, alpha=0.85, width=0.6,
                  error_kw={'linewidth': 1.5})

    # Shade offline vs online regions
    ax.axvspan(-0.5, 2.5, alpha=0.04, color='gray', label='Offline-z')
    ax.axvspan(2.5, 5.5, alpha=0.04, color='green', label='Online-z')
    ax.axhline(means[0], color=COLORS['MLP'], linestyle='--',
               linewidth=1.2, alpha=0.6, label='MLP baseline')

    ax.text(1.0, 0.97, 'Offline-z', transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color='gray')
    ax.text(0.55, 0.97, 'Online-z', transform=ax.transAxes,
            ha='left', va='top', fontsize=9, color='green')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Success rate (Scene3)', fontsize=11)
    ax.set_title('Online-z inference recovers and exceeds the offline MLP baseline', fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUT, 'fig7_headline_bar.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    fig.savefig(out.replace('.pdf','.png'), bbox_inches='tight', dpi=150)
    print(f'Saved {out}')
    plt.close(fig)


# ── Fig 8: Offline attention penalty + online recovery (paired) ───────────────
def fig_paired_comparison():
    """
    Shows the core finding as a delta from MLP baseline:
    offline attention hurts, online-z restores it.
    """
    mlp_mean = np.mean([0.60, 0.84])

    deltas = {
        'MoG\n(offline)':    np.mean([0.70, 0.68]) - mlp_mean,
        'Att-w4\n(offline)': np.mean([0.68, 0.56]) - mlp_mean,
        'OZ-MLP\n(online)':  np.mean([0.46, 0.74]) - mlp_mean,
        'OZ-Att-w4\n(online)': np.mean([0.88, 0.78]) - mlp_mean,
        'OZ-Att-w8\n(online)': np.mean([0.84, 0.80]) - mlp_mean,
    }
    delta_colors = ['#ff7f0e', '#d62728', '#17becf', '#2ca02c', '#9467bd']

    labels = list(deltas.keys())
    vals   = list(deltas.values())

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=delta_colors, alpha=0.85, width=0.55)
    ax.axhline(0, color='black', linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Δ success vs. MLP baseline', fontsize=11)
    ax.set_title('Change in success rate relative to MLP baseline (Scene3, 2 seeds)',
                 fontsize=10)
    ax.axvspan(1.5, 4.5, alpha=0.05, color='green')
    ax.text(0.62, 0.95, 'Online-z', transform=ax.transAxes,
            fontsize=9, color='green', va='top')
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + (0.01 if val >= 0 else -0.025),
                f'{val:+.2f}', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=9, fontweight='bold')
    plt.tight_layout()

    out = os.path.join(OUT, 'fig8_delta_bar.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    fig.savefig(out.replace('.pdf','.png'), bbox_inches='tight', dpi=150)
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    os.chdir('/scratch/gpfs/HASSON/ab4736/modified-infom')
    fig_pretraining()
    fig_tsne_phase()   # skips gracefully if no cache
    fig_headline_bar()
    fig_paired_comparison()
    print('Done.')
