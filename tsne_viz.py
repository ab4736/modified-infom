"""
t-SNE visualization of latent intentions learned by InFOM encoders.

Loads pretrained checkpoints (MLP and Attention) for scene3, runs the
intention encoder over the pretraining dataset, then plots t-SNE colored
by time-within-trajectory (proxy for behavioral phase).

Usage:
    python tsne_viz.py

Output: paper/tsne_intentions.pdf  (and .png)
"""

import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
import jax
import jax.numpy as jnp

# ── project imports ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset
from utils.flax_utils import restore_agent
from agents.infom import InFOMAgent

# ── config ───────────────────────────────────────────────────────────────────

SCENE3_ENV = 'scene-play-singletask-task3-v0'

# Each entry: (label, checkpoint_dir, flags_json_path, encoder_type)
RUNS = [
    (
        'MLP',
        'exp/scene3_mlp_seed0/sd000_s_7633776.0.20260503_113221',
        'exp/scene3_mlp_seed0/sd000_s_7633776.0.20260503_113221/flags.json',
        'mlp',
    ),
    (
        'Attention (w=4)',
        'exp/scene3_attention_seed0/sd000_s_7633778.0.20260503_120218',
        'exp/scene3_attention_seed0/sd000_s_7633778.0.20260503_120218/flags.json',
        'attention',
    ),
]

# How many dataset transitions to embed (keep low for speed on login node)
N_SAMPLES = 4000
# t-SNE hyperparams
TSNE_PERPLEXITY = 40
TSNE_N_ITER = 1000
RANDOM_SEED = 42

# Checkpoint epoch that was saved (see params_*.pkl in the run dir)
CHECKPOINT_EPOCH = 1500000

OUTPUT_DIR = 'paper'

# ── helpers ──────────────────────────────────────────────────────────────────

def load_flags(flags_path):
    with open(flags_path) as f:
        return json.load(f)


def build_agent_from_flags(flags):
    """Instantiate an InFOMAgent using the same flags used during training."""
    config = flags['agent']
    env_name = flags['env_name']
    frame_stack = flags.get('frame_stack', None)

    # Build dataset just to get example obs/action shapes
    _, _, dataset, _ = make_env_and_datasets(
        env_name, frame_stack=frame_stack, max_size=500, reward_free=True)
    dataset = Dataset.create(**dataset)
    dataset.frame_stack = frame_stack
    if config['encoder_type'] in ['attention', 'boundary_attention']:
        dataset.window_size = config['window_size']
    dataset.normalize_observations()

    example_batch = dataset.sample(1)

    agent = InFOMAgent.create(
        seed=0,
        ex_observations=example_batch['observations'],
        ex_actions=example_batch['actions'],
        config=config,
    )
    return agent, dataset, config


def collect_z_vectors(agent, dataset, config, n_samples, seed=RANDOM_SEED):
    """
    Sample transitions from the dataset and run the intention encoder.

    Returns:
        z_arr:   (n_samples, latent_dim) float32 array of intention means
        phase:   (n_samples,) float in [0,1] — normalized position within traj
        traj_id: (n_samples,) int — trajectory index for coloring
    """
    rng = np.random.default_rng(seed)

    # Collect raw indices ── we want contiguous windows so we can build
    # window_obs for the attention encoder.
    window_size = config.get('window_size', 1)
    encoder_type = config['encoder_type']

    # Sample random valid indices (not too close to trajectory boundaries)
    all_idxs = np.arange(len(dataset['observations']))
    # Mask out the last step of each trajectory to avoid cross-traj windows
    valid_mask = np.ones(len(dataset['observations']), dtype=bool)
    # Mark trajectory boundaries using dones / timeouts
    if 'masks' in dataset:
        done_idxs = np.where(dataset['masks'] == 0)[0]
    else:
        # Fall back: mark nothing as invalid (slight cross-traj bleed OK for viz)
        done_idxs = np.array([], dtype=int)
    for d in done_idxs:
        # Exclude window_size steps before each done
        lo = max(0, d - window_size + 1)
        valid_mask[lo:d + 1] = False

    valid_idxs = all_idxs[valid_mask]
    chosen = rng.choice(valid_idxs, size=min(n_samples, len(valid_idxs)),
                        replace=False)
    chosen = np.sort(chosen)

    z_list = []
    phase_list = []

    # Estimate trajectory start/end by scanning masks
    # Build a simple traj_id array
    traj_ids = np.zeros(len(dataset['observations']), dtype=int)
    tid = 0
    traj_start = np.zeros(len(dataset['observations']), dtype=int)
    traj_len_arr = np.zeros(len(dataset['observations']), dtype=int)
    cur_start = 0
    for i in range(len(dataset['observations'])):
        traj_ids[i] = tid
        traj_start[i] = cur_start
        if 'masks' in dataset and dataset['masks'][i] == 0:
            traj_len = i - cur_start + 1
            traj_len_arr[cur_start:i + 1] = traj_len
            tid += 1
            cur_start = i + 1
    # Handle last traj
    if cur_start < len(dataset['observations']):
        traj_len = len(dataset['observations']) - cur_start
        traj_len_arr[cur_start:] = traj_len

    print(f'Collecting {len(chosen)} z vectors '
          f'(encoder_type={encoder_type}, window={window_size})...')

    BATCH = 256
    for start in range(0, len(chosen), BATCH):
        batch_idxs = chosen[start:start + BATCH]

        if encoder_type in ['attention', 'boundary_attention']:
            # Build window: for each idx collect [idx-K+1 .. idx]
            # Clamp to traj_start to avoid crossing boundaries
            K = window_size
            win_obs = []
            win_acts = []
            for idx in batch_idxs:
                t_start = traj_start[idx]
                w_obs = []
                w_acts = []
                for offset in reversed(range(K)):
                    src = max(idx - offset, t_start)
                    w_obs.append(dataset['observations'][src])
                    w_acts.append(dataset['actions'][src])
                win_obs.append(np.stack(w_obs, axis=0))   # (K, obs_dim)
                win_acts.append(np.stack(w_acts, axis=0)) # (K, act_dim)
            win_obs = np.stack(win_obs, axis=0)   # (B, K, obs_dim)
            win_acts = np.stack(win_acts, axis=0) # (B, K, act_dim)
            dist = agent.network.select('intention_encoder')(win_obs, win_acts)
            z = np.array(dist.mean())  # (B, latent_dim)
        else:
            # MLP: single step (s', a')
            obs = dataset['next_observations'][batch_idxs]   # (B, obs_dim)
            acts = dataset['actions'][batch_idxs]             # (B, act_dim)
            dist = agent.network.select('intention_encoder')(obs, acts)
            z = np.array(dist.mean())  # (B, latent_dim)

        z_list.append(np.array(z))

        # Normalized position within trajectory [0, 1]
        ph = (batch_idxs - traj_start[batch_idxs]) / np.maximum(
            traj_len_arr[batch_idxs] - 1, 1)
        phase_list.append(ph)

    z_arr = np.concatenate(z_list, axis=0)
    phase_arr = np.concatenate(phase_list, axis=0)
    return z_arr, phase_arr, traj_ids[chosen]


def run_tsne(z_arr):
    print(f'  Running t-SNE on {z_arr.shape[0]} points, dim={z_arr.shape[1]}...')
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        max_iter=TSNE_N_ITER,
        random_state=RANDOM_SEED,
    )
    return tsne.fit_transform(z_arr.astype(np.float64))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_runs = len(RUNS)
    fig, axes = plt.subplots(1, n_runs, figsize=(5.5 * n_runs, 4.5))
    if n_runs == 1:
        axes = [axes]

    cmap = plt.cm.plasma

    for ax, (label, ckpt_dir, flags_path, enc_type) in zip(axes, RUNS):
        print(f'\n=== {label} ===')

        # ── load agent ──
        flags = load_flags(flags_path)
        agent, dataset, config = build_agent_from_flags(flags)

        ckpt_path = os.path.join(ckpt_dir, f'params_{CHECKPOINT_EPOCH}.pkl')
        if not os.path.exists(ckpt_path):
            # Try to find any available checkpoint
            import glob as _glob
            candidates = _glob.glob(os.path.join(ckpt_dir, 'params_*.pkl'))
            if not candidates:
                print(f'  WARNING: no checkpoint found in {ckpt_dir}, skipping.')
                ax.text(0.5, 0.5, 'checkpoint\nnot found',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12, color='gray')
                ax.set_title(label, fontsize=13, fontweight='bold')
                continue
            ckpt_path = sorted(candidates)[-1]
            print(f'  Using checkpoint: {ckpt_path}')

        agent = restore_agent(agent, os.path.dirname(ckpt_path),
                              int(os.path.basename(ckpt_path)
                                  .replace('params_', '').replace('.pkl', '')))

        # ── collect z ──
        z_arr, phase_arr, _ = collect_z_vectors(
            agent, dataset, config, N_SAMPLES)

        # ── t-SNE ──
        emb = run_tsne(z_arr)

        # ── plot ──
        sc = ax.scatter(
            emb[:, 0], emb[:, 1],
            c=phase_arr, cmap=cmap,
            s=6, alpha=0.7, linewidths=0,
            vmin=0, vmax=1,
        )
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xlabel('t-SNE dim 0', fontsize=10)
        ax.set_ylabel('t-SNE dim 1', fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_aspect('equal', adjustable='datalim')

    # shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label('Position within trajectory', fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['start', 'mid', 'end'])

    fig.suptitle(
        'Latent intention representations (scene3, t-SNE)',
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    pdf_path = os.path.join(OUTPUT_DIR, 'tsne_intentions.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'tsne_intentions.png')
    fig.savefig(pdf_path, bbox_inches='tight', dpi=150)
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    print(f'\nSaved: {pdf_path}')
    print(f'Saved: {png_path}')


if __name__ == '__main__':
    main()
