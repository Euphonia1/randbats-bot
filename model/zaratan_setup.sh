#!/bin/bash
# Make the Zaratan venv model/zaratan_train.sbatch trains in: CUDA PyTorch and
# CUDA JAX. Run once, on a login node (it needs the internet, which compute
# nodes may not have), e.g. from WSL:
#
#   ssh zaratan bash -s < model/zaratan_setup.sh
#
# Everything large goes to scratch: the home quota is about 10 GB, and PyTorch
# with its CUDA libraries is half of that. The cluster's Python stops at 3.10,
# so uv brings its own.
set -euo pipefail
REPO=${REPO:-$HOME/randbats-bot}
SCRATCH=${SCRATCH:-$HOME/scratch}
VENV=${VENV:-$SCRATCH/randbats-train}
export UV_CACHE_DIR=$SCRATCH/uv-cache UV_PYTHON_INSTALL_DIR=$SCRATCH/uv-python
UV=$HOME/.local/bin/uv

[ -x "$UV" ] || curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
"$UV" venv --allow-existing --python 3.14 "$VENV"
# CUDA 12 builds, which run on any driver from 525 up; CUDA 13 needs 580.
"$UV" pip install -p "$VENV" torch "jax[cuda12]==0.11.2" pytest \
    --extra-index-url https://download.pytorch.org/whl/cu128
# The repo's packages by path, so the checkout stays a plain checkout
SP=$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
printf '%s\n' "$REPO/simulator" "$REPO/model" > "$SP/randbats.pth"

"$VENV/bin/python" -c "
import jax, torch, psjax, architechture
print('jax', jax.__version__, '| torch', torch.__version__, 'CUDA', torch.version.cuda)"
