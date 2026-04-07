# ManipArena pi0.5 Training Pipeline

This pipeline targets an Ubuntu GPU machine with conda. It fine-tunes `lerobot/pi05_base` on the 15 ManipArena tabletop tasks, then serves the trained checkpoint through this repo's WebSocket protocol.

## 1. Environment

```bash
bash training/setup_pi05_env.sh
conda activate maniparena-pi05
```

## 2. Download Official Data

```bash
huggingface-cli login
huggingface-cli download ManipArena/maniparena-dataset \
  --repo-type dataset \
  --local-dir data/maniparena-dataset \
  --include "real/execution_reasoning/**" "real/semantic_reasoning/**"
```

The conversion script first looks for final files under `data/maniparena-dataset/real/...`. If that directory is absent, it also supports Hugging Face's local cache layout under `data/maniparena-dataset/.cache/huggingface/download/real/...`. `.lock` files alone are not usable data; if only locks exist, wait for the download to finish or rerun the download command.

## 3. Build the Tabletop Dataset

```bash
python training/prepare_maniparena_tabletop15.py \
  --source data/maniparena-dataset \
  --repo-id maniparena/tabletop15_pi05 \
  --root data/lerobot/maniparena_tabletop15_pi05 \
  --fps 20
```

## 4. Fine-tune pi0.5

```bash
bash training/train_pi05_tabletop15.sh smoke
bash training/train_pi05_tabletop15.sh main
```

Use `extended` for a longer run:

```bash
bash training/train_pi05_tabletop15.sh extended
```

Useful overrides:

```bash
BATCH_SIZE=8 STEPS=20000 TRAIN_EXPERT_ONLY=false bash training/train_pi05_tabletop15.sh main
```

## 5. Serve the Checkpoint

```bash
bash training/serve_pi05_tabletop15.sh outputs/pi05_tabletop15/checkpoints/last/pretrained_model
```

## 6. Local Checks

In another shell:

```bash
python scripts/mock_ping.py --uri ws://127.0.0.1:8000
python scripts/mock_schema_check.py --uri ws://127.0.0.1:8000
python scripts/eval_openloop.py \
  --server ws://127.0.0.1:8000 \
  --dataset data/maniparena-dataset/real/semantic_reasoning/press_button_in_order \
  --episode 0 \
  --save-dir openloop_plots \
  --action-chunk 32
```

These checks do not replace the official real-robot leaderboard evaluation.
