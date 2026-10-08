#!/bin/bash
# Train on the GPU under WSL, in the venv model/wsl_setup.sh makes.
#
#   wsl -d Ubuntu -- bash /mnt/c/Users/alexj/randbats-bot/model/wsl_train.sh [train.py arguments]
#
# With no arguments it carries on with runs/selfplay. It runs in the
# foreground and also writes everything to <run dir>/train.log, so to keep it
# going after the terminal closes, start it detached from PowerShell:
#
#   Start-Process wsl -WindowStyle Hidden -ArgumentList '-d','Ubuntu','--','bash','/mnt/c/Users/alexj/randbats-bot/model/wsl_train.sh'
#
# Follow it:  wsl -d Ubuntu -- tail -f /mnt/c/Users/alexj/randbats-bot/runs/selfplay/train.log
# Stop it:    wsl -d Ubuntu -- pkill -INT -f model/train.py
#             (an interrupt saves a checkpoint on the way out)
set -euo pipefail
VENV=${VENV:-~/randbats-train}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

[ $# -eq 0 ] && set -- --run-dir runs/selfplay --resume
run_dir=runs/selfplay
args=("$@")
for i in "${!args[@]}"; do
  [ "${args[$i]}" = "--run-dir" ] && run_dir=${args[$((i + 1))]}
done
mkdir -p "$run_dir"

# Compiled programs go to disk, so a restart loads them instead of recompiling.
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-~/.cache/jax}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

echo "=== $(date '+%F %T') train.py $*" >> "$run_dir/train.log"
"$VENV/bin/python" model/train.py "$@" 2>&1 | tee -a "$run_dir/train.log"
