#!/usr/bin/env bash
set -euo pipefail

# Ultra-High-Quality MuseTalk Offline Inference Script
# Targets: webcam.mp4 + webcam.mp3
# Optimized for: detail preservation and stable blending

BASE_DIR="/teamspace/studios/this_studio/onebox-deploy_fixed"
MUSE_CODE_DIR="/teamspace/studios/this_studio/MuseTalk"
cd "${BASE_DIR}"

# Create results directory if it doesn't exist
mkdir -p results/offline_best_quality

echo "Starting Ultra-High-Quality MuseTalk Offline Inference..."
# We mount the local MuseTalk folder to /opt/musetalk to apply our quality edits
docker run --rm \
  --gpus all \
  --env-file .env \
  --entrypoint python3 \
  -w /opt/musetalk \
  -v "${MUSE_CODE_DIR}:/opt/musetalk" \
  -v "${BASE_DIR}/models:/opt/musetalk/models" \
  -v "${BASE_DIR}/results:/opt/musetalk/results" \
  -v "${BASE_DIR}/.cache:/root/.cache" \
  -v "${BASE_DIR}/webcam.mp4:/opt/musetalk/input/webcam.mp4" \
  -v "${BASE_DIR}/webcam.mp3:/opt/musetalk/input/webcam.mp3" \
  -v "${BASE_DIR}/offline_inference.yaml:/opt/musetalk/configs/inference/offline.yaml" \
  coldslim/musetalk-onebox:blackwell \
  -m scripts.inference \
    --inference_config configs/inference/offline.yaml \
    --unet_model_path ./models/musetalkV15/unet.pth \
    --unet_config ./models/musetalkV15/musetalk.json \
    --version v15 \
    --parsing_mode jaw \
    --result_dir /opt/musetalk/results/offline_best_quality
