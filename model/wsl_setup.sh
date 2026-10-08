#!/bin/bash
# Make the WSL venv model/wsl_train.sh trains in: CUDA PyTorch and CUDA JAX
# side by side. Run once, from Windows:
#
#   wsl -d Ubuntu -- bash /mnt/c/Users/alexj/randbats-bot/model/wsl_setup.sh
#
# A venv of its own rather than ~/psjax-gpu (the simulator's JAX-only one):
# PyTorch pins older CUDA libraries than that venv has. They all still meet
# JAX's minimums, so here pip installs PyTorch's and JAX runs on them.
set -euo pipefail
VENV=${VENV:-~/randbats-train}
REPO=$(cd "$(dirname "$0")/.." && pwd)

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu130
"$VENV/bin/pip" install "jax[cuda13]==0.11.2" pytest
# The repo's packages by path, so nothing is written into the Windows checkout
SP=$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
printf '%s\n' "$REPO/simulator" "$REPO/model" > "$SP/randbats.pth"

"$VENV/bin/python" - <<'EOF'
import jax, torch
print("jax", jax.__version__, jax.default_backend(), jax.devices())
print("torch", torch.__version__, "cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
import psjax, architechture  # noqa: F401
EOF
