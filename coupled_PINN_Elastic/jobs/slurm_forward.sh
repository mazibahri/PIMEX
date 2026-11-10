#!/bin/bash
# Forward runs on a SLURM cluster: 2 stages x 3 seeds.
# Array index: even -> stage 1, odd -> stage 2; seed = index / 2.
# Edit the partition / GPU lines for your cluster, then:  sbatch jobs/slurm_forward.sh
#SBATCH --job-name=b01_fwd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=runs/slurm_%A_%a.out

export OMP_NUM_THREADS=4
PY=${PYTHON:-python}
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
STAGE=$(( SLURM_ARRAY_TASK_ID % 2 + 1 ))
SEED=$(( SLURM_ARRAY_TASK_ID / 2 ))
W_PLATE=${W_PLATE:-8.4}

$PY train.py --stage $STAGE --seed $SEED --adam 10000 --lbfgs 10000 --threads 4 \
    --w_plate $W_PLATE --out runs/s${STAGE}_seed${SEED}
