#!/bin/bash
# Run from login node: bash submit_scene_windows.sh
#
# Jobs (no data generation needed — scene data already exists):
#   JOB1: Attention w=8  on scene task3 seed 0
#   JOB2: Attention w=16 on scene task3 seed 0
#   JOB3: Attention w=32 on scene task3 seed 0

PYTHON="/scratch/gpfs/HASSON/ab4736/conda/envs/infom-sanity/bin/python"
WORK="/scratch/gpfs/HASSON/ab4736/modified-infom"

mkdir -p $WORK/exp/scene3_att_w8 \
         $WORK/exp/scene3_att_w16 \
         $WORK/exp/scene3_att_w32

TRAIN_FLAGS="--env_name=scene-play-singletask-task3-v0 --pretraining_steps=1_000_000 --finetuning_steps=500_000 --eval_interval=50_000 --save_interval=9_999_999 --seed=0 --enable_wandb=0 --agent.latent_dim=128 --agent.expectile=0.99 --agent.kl_weight=0.2 --agent.alpha=300 --agent.encoder_type=attention"

# ── JOB1: Attention w=8 ───────────────────────────────────────────────────────
JOB1=$(sbatch --parsable <<SLURM
#!/bin/bash
#SBATCH --job-name=scene3_att_w8
#SBATCH --output=$WORK/exp/scene3_att_w8/%j.out
#SBATCH --error=$WORK/exp/scene3_att_w8/%j.err
#SBATCH --time=2:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

export PYTHONPATH=$WORK

echo "=== Run: scene3 Attention w=8 ==="
$PYTHON $WORK/main.py \
    $TRAIN_FLAGS \
    --save_dir=$WORK/exp/scene3_att_w8 \
    --wandb_run_group=scene3_att_w8 \
    --agent.window_size=8
echo "=== scene3 Attention w=8 DONE ==="
SLURM
)
echo "Submitted JOB1 (scene3 Attention w=8): $JOB1"

# ── JOB2: Attention w=16 ──────────────────────────────────────────────────────
JOB2=$(sbatch --parsable <<SLURM
#!/bin/bash
#SBATCH --job-name=scene3_att_w16
#SBATCH --output=$WORK/exp/scene3_att_w16/%j.out
#SBATCH --error=$WORK/exp/scene3_att_w16/%j.err
#SBATCH --time=2:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

export PYTHONPATH=$WORK

echo "=== Run: scene3 Attention w=16 ==="
$PYTHON $WORK/main.py \
    $TRAIN_FLAGS \
    --save_dir=$WORK/exp/scene3_att_w16 \
    --wandb_run_group=scene3_att_w16 \
    --agent.window_size=16
echo "=== scene3 Attention w=16 DONE ==="
SLURM
)
echo "Submitted JOB2 (scene3 Attention w=16): $JOB2"

# ── JOB3: Attention w=32 ──────────────────────────────────────────────────────
JOB3=$(sbatch --parsable <<SLURM
#!/bin/bash
#SBATCH --job-name=scene3_att_w32
#SBATCH --output=$WORK/exp/scene3_att_w32/%j.out
#SBATCH --error=$WORK/exp/scene3_att_w32/%j.err
#SBATCH --time=2:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

export PYTHONPATH=$WORK

echo "=== Run: scene3 Attention w=32 ==="
$PYTHON $WORK/main.py \
    $TRAIN_FLAGS \
    --save_dir=$WORK/exp/scene3_att_w32 \
    --wandb_run_group=scene3_att_w32 \
    --agent.window_size=32
echo "=== scene3 Attention w=32 DONE ==="
SLURM
)
echo "Submitted JOB3 (scene3 Attention w=32): $JOB3"

echo ""
echo "Pipeline:"
echo "  JOB1 ($JOB1): scene3 Attention w=8   [2.5h]"
echo "  JOB2 ($JOB2): scene3 Attention w=16  [2.5h]"
echo "  JOB3 ($JOB3): scene3 Attention w=32  [2.5h]"
echo ""
echo "Monitor: squeue -u $USER"
