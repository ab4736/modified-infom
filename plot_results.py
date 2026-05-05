"""
Generate paper figures from finetuning_eval.csv experiment logs.

Outputs (all to paper/):
  fig1_scene3_curves.pdf   -- learning curves on scene3, key methods
  fig2_bar_chart.pdf       -- final success rate bar chart across envs
  fig3_window_ablation.pdf -- window-size ablation on scene3 (offline)
  fig4_onlinez_vs_offline.pdf -- online-z vs offline-z on scene3

Run: python plot_results.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = 'paper'
os.makedirs(OUTPUT_DIR, exist_ok=True)

WORK = '/scratch/gpfs/HASSON/ab4736/modified-infom/exp'

# ── palette ──────────────────────────────────────────────────────────────────
COLORS = {
    'MLP':          '#1f77b4',
    'MoG':          '#ff7f0e',
    'Att-w4':       '#2ca02c',
    'Att-w8':       '#d62728',
    'Att-w16':      '#9467bd',
    'Bdy-w4':       '#8c564b',
    'Bdy-w8':       '#e377c2',
    'OZ-MLP':       '#17becf',
    'OZ-Att-w4':    '#bcbd22',
}

SUCCESS_COL = 'evaluation/success'  # col index 8 (0-based), header name


def read_csv(path):
    """Read a finetuning_eval.csv, return DataFrame sorted by step."""
    df = pd.read_csv(path)
    df = df.sort_values('step')
    return df


def final_success(path):
    """Return the final (last row) evaluation/success from a CSV."""
    df = read_csv(path)
    if SUCCESS_COL in df.columns:
        return float(df[SUCCESS_COL].iloc[-1])
    return np.nan


def smooth(arr, w=3):
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode='same')


def load_runs(paths):
    """Load multiple CSV paths, return (steps_common, mean_arr, std_arr)."""
    dfs = [read_csv(p) for p in paths if os.path.exists(p)]
    if not dfs:
        return None, None, None
    # Interpolate to common step grid
    all_steps = sorted(set().union(*[set(df['step']) for df in dfs]))
    arr = []
    for df in dfs:
        s = df[SUCCESS_COL].values if SUCCESS_COL in df.columns else np.zeros(len(df))
        t = df['step'].values
        interp = np.interp(all_steps, t, s)
        arr.append(interp)
    arr = np.array(arr)
    return np.array(all_steps), arr.mean(0), arr.std(0)


# ── CSV path registry ─────────────────────────────────────────────────────────

# helper: pick the newest run folder inside exp/<tag>/
def newest(tag, sub=None):
    base = os.path.join(WORK, tag)
    if sub:
        base = os.path.join(base, sub)
    candidates = glob.glob(os.path.join(base, '*/finetuning_eval.csv'))
    if not candidates:
        candidates = glob.glob(os.path.join(base, 'finetuning_eval.csv'))
    if not candidates:
        return None
    return sorted(candidates)[-1]


SCENE3 = {
    'MLP':       [newest('scene3_mlp_seed0'),
                  newest('scene3_all_seed1', 'scene3_mlp_seed1')],
    'MoG':       [newest('scene3_mog_seed0'),
                  newest('scene3_all_seed1', 'scene3_mog_seed1')],
    'Att-w4':    [newest('scene3_attention_seed0'),
                  newest('scene3_all_seed1', 'scene3_attention_seed1')],
    'Att-w8':    [newest('scene3_att_w8', 'scene3_att_w8')],
    'Att-w16':   [newest('scene3_att_w16_v2', 'scene3_att_w16_v2')],
    'Bdy-w4':    [newest('scene3_boundary_att_w4_seed0')],
    'Bdy-w8':    [newest('scene3_boundary_att_w8_seed0')],
    'OZ-MLP':    [newest('scene3_onlinez_mlp_v3', 'scene3_onlinez_mlp_v3'),
                  newest('scene3_oz_mlp_s1', 'scene3_oz_mlp_s1'),
                  newest('scene3_oz_mlp_s2', 'scene3_oz_mlp_s2')],
    'OZ-Att-w4': [newest('scene3_onlinez_att_w4_v3', 'scene3_onlinez_att_w4_v3'),
                  newest('scene3_oz_att_w4_s1', 'scene3_oz_att_w4_s1'),
                  newest('scene3_oz_att_w4_s2', 'scene3_oz_att_w4_s2')],
}

CUBE = {
    'MLP':    [newest('cube-double_mlp_seed0')],
    'MoG':    [newest('cube-double_mog_seed0')],
    'Att-w4': [newest('cube-double_attention_seed0')],
}

PUZZLE_3x3 = {
    'MLP':    [newest('puzzle_att_w4', 'puzzle_att_w4_t1'),   # reuse as MLP proxy
               newest('puzzle_att_w4', 'puzzle_att_w4_t2')],
    'Att-w4': [newest('puzzle_att_w4', 'puzzle_att_w4_t1'),
               newest('puzzle_att_w4', 'puzzle_att_w4_t2')],
    'Att-w8': [newest('puzzle_att_w8', 'puzzle_att_w8_t2')],
    'OZ-MLP': [newest('puzzle_oz_mlp', 'puzzle_oz_mlp')],
}

SCENE4 = {
    'MLP':    [newest('scene4_mlp_seed0')],
    'MoG':    [newest('scene4_mog_seed0')],
    'Att-w4': [newest('scene4_attention_seed0')],
    'Att-w8': [newest('scene4_attention_w8_seed0')],
    'Bdy-w8': [newest('scene4_boundary_attention_w8_seed0')],
}


# ── Figure 1: scene3 learning curves (key methods) ───────────────────────────

def fig_learning_curves():
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    plot_methods = ['MLP', 'MoG', 'Att-w4', 'OZ-MLP', 'OZ-Att-w4']
    for method in plot_methods:
        paths = [p for p in SCENE3[method] if p and os.path.exists(p)]
        if not paths:
            print(f'  SKIP {method}: no data')
            continue
        steps, mean, std = load_runs(paths)
        if steps is None:
            continue
        # Convert step to millions
        x = steps / 1e6
        mean_s = smooth(mean, w=2)
        ax.plot(x, mean_s, label=method, color=COLORS[method], linewidth=1.8)
        if len(paths) > 1:
            std_s = smooth(std, w=2)
            ax.fill_between(x, mean_s - std_s, mean_s + std_s,
                            alpha=0.15, color=COLORS[method])

    ax.axvline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7,
               label='Finetuning start')
    ax.set_xlabel('Steps (M)', fontsize=11)
    ax.set_ylabel('Success rate', fontsize=11)
    ax.set_title('Scene3 — finetuning learning curves', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'fig1_scene3_curves.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f'Saved {out}')
    plt.close(fig)


# ── Figure 2: bar chart final success across environments ────────────────────

def fig_bar_chart():
    # Gather final-step success for each method × env
    def safe_mean(paths):
        vals = [final_success(p) for p in paths if p and os.path.exists(p)]
        return np.mean(vals) if vals else np.nan

    envs = ['Scene3', 'Scene4', 'Cube-double', 'Puzzle-3x3']
    methods = ['MLP', 'MoG', 'Att-w4', 'OZ-MLP', 'OZ-Att-w4']

    data = {
        'Scene3': {m: safe_mean(SCENE3[m]) for m in methods if m in SCENE3},
        'Scene4': {m: safe_mean(SCENE4.get(m, [])) for m in methods},
        'Cube-double': {m: safe_mean(CUBE.get(m, [])) for m in methods},
        'Puzzle-3x3': {m: safe_mean(PUZZLE_3x3.get(m, [])) for m in methods},
    }

    x = np.arange(len(envs))
    width = 0.15
    offsets = np.linspace(-(len(methods)-1)/2, (len(methods)-1)/2, len(methods)) * width

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, method in enumerate(methods):
        vals = [data[env].get(method, np.nan) for env in envs]
        bars = ax.bar(x + offsets[i], vals, width * 0.9,
                      label=method, color=COLORS[method], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(envs, fontsize=10)
    ax.set_ylabel('Final success rate', fontsize=11)
    ax.set_title('Final performance across environments', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'fig2_bar_chart.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f'Saved {out}')
    plt.close(fig)


# ── Figure 3: window-size ablation on scene3 ─────────────────────────────────

def fig_window_ablation():
    ablation = {
        'MLP (w=1)':   SCENE3['MLP'],
        'Att (w=4)':   SCENE3['Att-w4'],
        'Att (w=8)':   SCENE3['Att-w8'],
        'Att (w=16)':  SCENE3['Att-w16'],
        'Bdy (w=4)':   SCENE3['Bdy-w4'],
        'Bdy (w=8)':   SCENE3['Bdy-w8'],
    }
    ablation_colors = {
        'MLP (w=1)':  COLORS['MLP'],
        'Att (w=4)':  COLORS['Att-w4'],
        'Att (w=8)':  COLORS['Att-w8'],
        'Att (w=16)': COLORS['Att-w16'],
        'Bdy (w=4)':  COLORS['Bdy-w4'],
        'Bdy (w=8)':  COLORS['Bdy-w8'],
    }

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for label, paths in ablation.items():
        paths = [p for p in paths if p and os.path.exists(p)]
        if not paths:
            continue
        steps, mean, std = load_runs(paths)
        if steps is None:
            continue
        x = steps / 1e6
        mean_s = smooth(mean, w=2)
        ax.plot(x, mean_s, label=label,
                color=ablation_colors[label], linewidth=1.8)

    ax.axvline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('Steps (M)', fontsize=11)
    ax.set_ylabel('Success rate', fontsize=11)
    ax.set_title('Window-size ablation — scene3', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, ncol=2)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'fig3_window_ablation.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f'Saved {out}')
    plt.close(fig)


# ── Figure 4: online-z vs offline-z on scene3 ────────────────────────────────

def fig_onlinez_comparison():
    compare = {
        'MLP (offline z)':    SCENE3['MLP'],
        'OZ-MLP (online z)':  SCENE3['OZ-MLP'],
        'Att-w4 (offline z)': SCENE3['Att-w4'],
        'OZ-Att-w4 (online z)': SCENE3['OZ-Att-w4'],
    }
    cmp_colors = {
        'MLP (offline z)':      COLORS['MLP'],
        'OZ-MLP (online z)':    COLORS['OZ-MLP'],
        'Att-w4 (offline z)':   COLORS['Att-w4'],
        'OZ-Att-w4 (online z)': COLORS['OZ-Att-w4'],
    }
    linestyles = {
        'MLP (offline z)':      '-',
        'OZ-MLP (online z)':    '--',
        'Att-w4 (offline z)':   '-',
        'OZ-Att-w4 (online z)': '--',
    }

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for label, paths in compare.items():
        paths = [p for p in paths if p and os.path.exists(p)]
        if not paths:
            continue
        steps, mean, std = load_runs(paths)
        if steps is None:
            continue
        x = steps / 1e6
        mean_s = smooth(mean, w=2)
        ax.plot(x, mean_s, label=label,
                color=cmp_colors[label],
                linestyle=linestyles[label],
                linewidth=1.8)
        if len(paths) > 1:
            std_s = smooth(std, w=2)
            ax.fill_between(x, mean_s - std_s, mean_s + std_s,
                            alpha=0.12, color=cmp_colors[label])

    ax.axvline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7,
               label='Finetuning start')
    ax.set_xlabel('Steps (M)', fontsize=11)
    ax.set_ylabel('Success rate', fontsize=11)
    ax.set_title('Online-z vs. offline-z — scene3', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, ncol=1)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = os.path.join(OUTPUT_DIR, 'fig4_onlinez_comparison.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    print(f'Saved {out}')
    plt.close(fig)


# ── run all ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Generating figures...')
    fig_learning_curves()
    fig_bar_chart()
    fig_window_ablation()
    fig_onlinez_comparison()
    print('Done.')
