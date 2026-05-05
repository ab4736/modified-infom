#!/bin/bash
# Run from login node: bash submit_scene_onlinez.sh
#
# Jobs: actor conditioned on online-inferred z (use_actor_z=True)
#   JOB1: MLP  encoder + online z, scene task3 seed 0
#   JOB2: Attention w=4 + online z, scene task3 seed 0

PYTHON="/scratch/gpfs/HASSON/ab4736/conda/envs/infom-sanity/bin/python"
WORK="/scratch/gpfs/HASSON/ab4736/modified-infom"

mkdir -p $WORK/exp/scene3_onlinez_mlp \
         $WORK/exp/scene3_onlinez_att_w4

TRAIN_FLAGS="--env_name=scene-play-singletask-task3-v0 --pretraining_steps=1_000_000 --finetuning_steps=500_000 --eval_interval=50_000 --save_interval=9_999_999 --seed=0 --enable_wandb=0 --agent.latent_dim=128 --agent.expectile=0.99 --agent.kl_weight=0.2 --agent.alpha=300 --agent.use_actor_z=1"

# ── JOB1: MLP + online z ──────────────────────────────────────────────────────
JOB1=$(sbatch --parsable <<SLURM
#!/bin/bash
#SBATCH --job-name=scene3_oz_mlp
#SBATCH --output=$WORK/exp/scene3_onlinez_mlp/%j.out
#SBATCH --error=$WORK/exp/scene3_onlinez_mlp/%j.err
#SBATCH --time=2:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

export PYTHONPATH=$WORK

echo "=== Run: scene3 MLP + online z ==="
$PYTHON $WORK/main.py \
    $TRAIN_FLAGS \
    --save_dir=$WORK/exp/scene3_onlinez_mlp \
    --wandb_run_group=scene3_onlinez_mlp \
    --agent.encoder_type=mlp
echo "=== scene3 MLP + online z DONE ==="
SLURM
)
echo "Submitted JOB1 (scene3 MLP + online z): $JOB1"

# ── JOB2: Attention w=4 + online z ───────────────────────────────────────────
JOB2=$(sbatch --parsable <<SLURM
#!/bin/bash
#SBATCH --job-name=scene3_oz_att4
#SBATCH --output=$WORK/exp/scene3_onlinez_att_w4/%j.out
#SBATCH --error=$WORK/exp/scene3_onlinez_att_w4/%j.err
#SBATCH --time=2:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

export PYTHONPATH=$WORK

echo "=== Run: scene3 Attention w=4 + online z ==="
$PYTHON $WORK/main.py \
    $TRAIN_FLAGS \
    --save_dir=$WORK/exp/scene3_onlinez_att_w4 \
    --wandb_run_group=scene3_onlinez_att_w4 \
    --agent.encoder_type=attention \
    --agent.window_size=4
echo "=== scene3 Attention w=4 + online z DONE ==="
SLURM
)
echo "Submitted JOB2 (scene3 Attention w=4 + online z): $JOB2"

echo ""
echo "Pipeline:"
echo "  JOB1 ($JOB1): scene3 MLP + online z      [2.5h]"
echo "  JOB2 ($JOB2): scene3 Att w=4 + online z  [2.5h]"
echo ""
echo "Monitor: squeue -u $USER"
