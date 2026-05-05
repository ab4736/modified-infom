#!/bin/bash
# Run from login node: bash submit_scene.sh

PYTHON="/scratch/gpfs/HASSON/ab4736/conda/envs/infom-sanity/bin/python"
WORK="/scratch/gpfs/HASSON/ab4736/modified-infom"
DATA="$HOME/.ogbench/data"

# Download pre-training dataset on login node (has internet)
echo "Downloading scene pre-training dataset..."
$PYTHON -c "
from ogbench.utils import download_datasets
download_datasets(['scene-play-v0'], '$DATA')
"
$PYTHON -c "
import urllib.request, os
url = 'https://rail.eecs.berkeley.edu/datasets/ogbench/scene-play-v0-val.npz'
dst = os.path.expanduser('~/.ogbench/data/scene-play-v0-val.npz')
os.makedirs(os.path.dirname(dst), exist_ok=True)
if not os.path.exists(dst):
    print('Downloading', url)
    urllib.request.urlretrieve(url, dst)
    print('Done')
else:
    print('Val file already exists, skipping.')
"
echo "Download done."

mkdir -p $WORK/exp/scene_mlp $WORK/exp/scene_mog $WORK/exp/scene_att

# Job 1: generate ft data + run mlp
JOB1=$(sbatch --parsable <<SLURM
#!/bin/bash
#SBATCH --job-name=scene_gen_mlp
#SBATCH --output=$WORK/exp/scene_mlp/%j.out
#SBATCH --error=$WORK/exp/scene_mlp/%j.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=$WORK

echo "=== Generating scene ft datasets ==="
$PYTHON $WORK/data_gen_scripts/generate_ogbench_manispace.py \
    --env_name=scene-v0 \
    --save_path=$DATA/scene-play-ft-v0.npz \
    --num_episodes=500 --max_episode_steps=1001 --dataset_type=play

$PYTHON $WORK/data_gen_scripts/generate_ogbench_manispace.py \
    --env_name=scene-v0 \
    --save_path=$DATA/scene-play-ft-v0-val.npz \
    --num_episodes=50 --max_episode_steps=1001 --dataset_type=play

echo "=== Run: mlp ==="
$PYTHON $WORK/main.py \
    --env_name=scene-play-singletask-task1-v0 \
    --seed 0 \
    --enable_wandb 0 \
    --save_dir $WORK/exp/scene_mlp \
    --wandb_run_group scene_mlp_seed0 \
    --agent.encoder_type mlp \
    --agent.latent_dim 128 \
    --agent.expectile 0.99 \
    --agent.kl_weight 0.2 \
    --agent.alpha 300

echo "=== mlp DONE ==="
SLURM
)
echo "Submitted job 1 (generate + mlp): $JOB1"

# Job 2: mog — waits for job 1
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 <<SLURM
#!/bin/bash
#SBATCH --job-name=scene_mog
#SBATCH --output=$WORK/exp/scene_mog/%j.out
#SBATCH --error=$WORK/exp/scene_mog/%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=$WORK

echo "=== Run: mog ==="
$PYTHON $WORK/main.py \
    --env_name=scene-play-singletask-task1-v0 \
    --seed 0 \
    --enable_wandb 0 \
    --save_dir $WORK/exp/scene_mog \
    --wandb_run_group scene_mog_seed0 \
    --agent.encoder_type mog \
    --agent.latent_dim 128 \
    --agent.expectile 0.99 \
    --agent.kl_weight 0.2 \
    --agent.alpha 300

echo "=== mog DONE ==="
SLURM
)
echo "Submitted job 2 (mog, depends on $JOB1): $JOB2"

# Job 3: attention — waits for job 1
JOB3=$(sbatch --parsable --dependency=afterok:$JOB1 <<SLURM
#!/bin/bash
#SBATCH --job-name=scene_att
#SBATCH --output=$WORK/exp/scene_att/%j.out
#SBATCH --error=$WORK/exp/scene_att/%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=$WORK

echo "=== Run: attention ==="
$PYTHON $WORK/main.py \
    --env_name=scene-play-singletask-task1-v0 \
    --seed 0 \
    --enable_wandb 0 \
    --save_dir $WORK/exp/scene_att \
    --wandb_run_group scene_att_seed0 \
    --agent.encoder_type attention \
    --agent.latent_dim 128 \
    --agent.expectile 0.99 \
    --agent.kl_weight 0.2 \
    --agent.alpha 300

echo "=== attention DONE ==="
SLURM
)
echo "Submitted job 3 (attention, depends on $JOB1): $JOB3"

echo ""
echo "Monitor: squeue -u $USER"
echo "Logs:    tail -f $WORK/exp/scene_mlp/<jobid>.out"
