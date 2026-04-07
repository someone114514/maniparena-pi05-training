#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-maniparena-pi05}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
LEROBOT_DIR="${LEROBOT_DIR:-$HOME/src/lerobot}"
LEROBOT_REPO="${LEROBOT_REPO:-https://github.com/huggingface/lerobot.git}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"
conda install -y -c conda-forge "ffmpeg=7.1.1"
python -m pip install --upgrade pip setuptools wheel

mkdir -p "$(dirname "$LEROBOT_DIR")"
if [ ! -d "$LEROBOT_DIR/.git" ]; then
  git clone "$LEROBOT_REPO" "$LEROBOT_DIR"
else
  git -C "$LEROBOT_DIR" fetch --all --tags --prune
fi

python -m pip install -e "$LEROBOT_DIR[pi]"
python -m pip install -r requirements.txt

python - <<'PY'
import torch
print("python ok")
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

echo "Environment ready. Run: conda activate $ENV_NAME"
