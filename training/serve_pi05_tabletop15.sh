#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-outputs/pi05_tabletop15/checkpoints/last/pretrained_model}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
DEVICE="${DEVICE:-cuda:0}"
ACTION_HORIZON="${ACTION_HORIZON:-50}"

export MANIPARENA_PI05_POLICY="${MANIPARENA_PI05_POLICY:-lerobot}"

python serve.py \
  --checkpoint "$CHECKPOINT" \
  --control-mode end_pose \
  --action-horizon "$ACTION_HORIZON" \
  --device "$DEVICE" \
  --host "$HOST" \
  --port "$PORT"
