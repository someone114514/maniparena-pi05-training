#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-main}"

case "$MODE" in
  smoke) STEPS="${STEPS:-3000}" ;;
  main) STEPS="${STEPS:-10000}" ;;
  extended) STEPS="${STEPS:-30000}" ;;
  *) echo "Usage: $0 [smoke|main|extended]" >&2; exit 2 ;;
esac

DATASET_REPO_ID="${DATASET_REPO_ID:-maniparena/tabletop15_pi05}"
DATASET_ROOT="${DATASET_ROOT:-data/lerobot/maniparena_tabletop15_pi05}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/pi05_tabletop15}"
JOB_NAME="${JOB_NAME:-pi05_tabletop15_${MODE}}"
BASE_POLICY="${BASE_POLICY:-lerobot/pi05_base}"
POLICY_REPO_ID="${POLICY_REPO_ID:-maniparena/pi05_tabletop15}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TRAIN_EXPERT_ONLY="${TRAIN_EXPERT_ONLY:-true}"
FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-false}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
NORMALIZATION_MAPPING='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'
NORMALIZATION_MAPPING="${NORMALIZATION_MAPPING_OVERRIDE:-$NORMALIZATION_MAPPING}"

if ! command -v lerobot-train >/dev/null 2>&1; then
  echo "lerobot-train not found. Activate the maniparena-pi05 conda env first." >&2
  exit 1
fi

HELP="$(lerobot-train --help 2>&1 || true)"
EXTRA_ARGS=()

if grep -q -- "--policy.pretrained_path" <<<"$HELP"; then
  EXTRA_ARGS+=("--policy.pretrained_path=$BASE_POLICY")
else
  EXTRA_ARGS+=("--policy.path=$BASE_POLICY")
fi
if grep -q -- "--policy.repo_id" <<<"$HELP"; then
  EXTRA_ARGS+=("--policy.repo_id=$POLICY_REPO_ID")
fi
if grep -q -- "--policy.compile_model" <<<"$HELP"; then
  EXTRA_ARGS+=("--policy.compile_model=$COMPILE_MODEL")
fi
if grep -q -- "--policy.normalization_mapping" <<<"$HELP"; then
  EXTRA_ARGS+=("--policy.normalization_mapping=$NORMALIZATION_MAPPING")
fi

lerobot-train \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --policy.type=pi05 \
  "${EXTRA_ARGS[@]}" \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.gradient_checkpointing=true \
  --policy.train_expert_only="$TRAIN_EXPERT_ONLY" \
  --policy.freeze_vision_encoder="$FREEZE_VISION_ENCODER" \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS"
