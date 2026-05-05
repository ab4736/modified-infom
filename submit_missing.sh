#!/bin/bash
# Submit OZ-MLP on cube-double and MoG on puzzle

PYTHON="/scratch/gpfs/HASSON/ab4736/conda/envs/infom-sanity/bin/python"
WORK="/scratch/gpfs/HASSON/ab4736/modified-infom"

mkdir -p $WORK/exp/cube-double_oz_mlp $WORK/exp/puzzle_mog

# Job 1: cube-double OZ-MLP
JOB1=$(sbatch --parsable <<SLURM
#!/bin/bash
#SBATCH --job-name=cube_oz_mlp
#SBATCH --output=$WORK/exp/cube-double_oz_mlp/%j.out
#SBATCH --error=$WORK/exp/cube-double_oz_mlp/%j.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=$WORK
source ~/.bashrc
conda activate /scratch/gpfs/HASSON/ab4736/conda/envs/infom-sanity

echo "=== cube-double OZ-MLP ==="
$PYTHON $WORK/main.py \
    --env_name=cube-double-play-singletask-task1-v0 \
    --seed 0 \
    --enable_wandb 0 \
    --save_dir $WORK/exp/cube-double_oz_mlp \
    --wandb_run_group cube_double_oz_mlp \
    --agent.encoder_type mlp \
    --agent.use_actor_z True \
    --agent.latent_dim 128 \
    --agent.expectile 0.9 \
    --agent.kl_weight 0.025 \
    --agent.alpha 30
echo "=== DONE ==="
SLURM
)
echo "Submitted cube-double OZ-MLP: $JOB1"

# Job 2: puzzle MoG
JOB2=$(sbatch --parsable <<SLURM
#!/bin/bash
#SBATCH --job-name=puzzle_mog
#SBATCH --output=$WORK/exp/puzzle_mog/%j.out
#SBATCH --error=$WORK/exp/puzzle_mog/%j.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=$WORK
source ~/.bashrc
conda activate /scratch/gpfs/HASSON/ab4736/conda/envs/infom-sanity

echo "=== puzzle MoG ==="
$PYTHON $WORK/main.py \
    --env_name=puzzle-4x4-play-singletask-task1-v0 \
    --seed 0 \
    --enable_wandb 0 \
    --save_dir $WORK/exp/puzzle_mog \
    --wandb_run_group puzzle_mog \
    --agent.encoder_type mog \
    --agent.num_components 3 \
    --agent.latent_dim 128 \
    --agent.expectile 0.95 \
    --agent.kl_weight 0.1 \
    --agent.alpha 300 \
    --pretraining_steps 1000000 \
    --finetuning_steps 500000
echo "=== DONE ==="
SLURM
)
echo "Submitted puzzle MoG: $JOB2"

echo ""
echo "Monitor: squeue -u $USER"
