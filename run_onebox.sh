#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

IMAGE_TAG="${IMAGE_TAG:-coldslim/musetalk-onebox:blackwell}"
ENV_FILE="${ENV_FILE:-${BASE_DIR}/.env}"
CONTAINER_NAME="${CONTAINER_NAME:-musetalk-onebox}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  echo "Create it from ${BASE_DIR}/env.example" >&2
  exit 1
fi

mkdir -p "${BASE_DIR}/models" "${BASE_DIR}/results" "${BASE_DIR}/.cache" "${BASE_DIR}/voices"
mkdir -p "${BASE_DIR}/turn-native"

echo "Starting container in headless mode..."
docker run --rm \
  --gpus all \
  --env-file "${ENV_FILE}" \
  -p 8780:8780 \
  -v "${BASE_DIR}/models:/opt/musetalk/models" \
  -v "${BASE_DIR}/results:/opt/musetalk/results" \
  -v "${BASE_DIR}/.cache:/root/.cache" \
  -v "${BASE_DIR}/voices:/opt/personaplex/custom_voices" \
  -v "${BASE_DIR}/turn-native:/opt/onebox/turn-native:ro" \
  "${IMAGE_TAG}"
