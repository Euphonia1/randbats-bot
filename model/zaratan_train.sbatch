#!/bin/bash
# Self-play training on one Zaratan GPU, in the venv model/zaratan_setup.sh
# makes. Submit from the repo on a login node:
#
#   cd ~/randbats-bot && sbatch model/zaratan_train.sbatch [train.py arguments]
#
# With no arguments it trains runs/zaratan, carrying on from its checkpoint
# when there is one, so resubmitting picks up where the last job stopped.
# Override the GPU or time limit on the command line:
#
#   sbatch --gpus=h100:1 --time=12:00:00 model/zaratan_train.sbatch
#
# Service units an hour: 48 for an A100 or 144 for an H100, plus 1 per CPU
# core and 0.25 per GB of memory -- 72 or 168 as set here.
#
# Follow it:  tail -f runs/zaratan/train.log
# Stop it:    scancel --full --signal=INT <job id>   (saves a checkpoint first)
#
#SBATCH --job-name=randbats-train
#SBATCH --account=msml612-class
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm-%j.out
# An interrupt two minutes before the time limit, so train.py saves
#SBATCH --signal=B:INT@120
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
VENV=${VENV:-$HOME/scratch/randbats-train}

[ $# -eq 0 ] && set -- --run-dir runs/zaratan --num-envs 1024
run_dir=runs/zaratan resume=0
args=("$@")
for i in "${!args[@]}"; do
  [ "${args[$i]}" = "--run-dir" ] && run_dir=${args[$((i + 1))]}
  [ "${args[$i]}" = "--resume" ] && resume=1
done
mkdir -p "$run_dir"
[ $resume = 0 ] && [ -f "$run_dir/checkpoint.pt" ] && set -- "$@" --resume

# Compiled programs go to disk, so a resubmitted job loads them instead of
# recompiling.
export JAX_COMPILATION_CACHE_DIR=$HOME/scratch/jax-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

{
  echo "=== $(date '+%F %T') job $SLURM_JOB_ID on $(hostname): train.py $*"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} >> "$run_dir/train.log"
# exec, so the time-limit interrupt reaches train.py itself
exec "$VENV/bin/python" model/train.py "$@" >> "$run_dir/train.log" 2>&1
