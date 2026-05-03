#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

IMAGE_TAG="${IMAGE_TAG:-coldslim/musetalk-onebox:blackwell}"
ENV_FILE="${ENV_FILE:-${BASE_DIR}/.env}"
CONTAINER_NAME="${CONTAINER_NAME:-musetalk-onebox}"
MUSE_WEBRTC_OVERRIDE_DIR="${MUSE_WEBRTC_OVERRIDE_DIR:-$(cd "${BASE_DIR}/.." && pwd)/MuseTalk/scripts/musetalk_webrtc}"
PERSONAPLEX_OVERRIDE_DIR="${PERSONAPLEX_OVERRIDE_DIR:-$(cd "${BASE_DIR}/.." && pwd)/personaplex/moshi/moshi}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  echo "Create it from ${BASE_DIR}/env.example" >&2
  exit 1
fi

echo "Using env file: ${ENV_FILE}"
echo "Realtime settings from env:"
grep -E '^(MUSE_FPS|MUSE_BATCH_SIZE|MUSE_WINDOW_MS|MUSE_HOP_MS|MUSE_MIN_WINDOW_MS|MUSE_MAX_ADVANCE_MS|MUSE_MAX_TAIL_FRAMES|MUSE_VIDEO_QUEUE_SIZE)=' "${ENV_FILE}" || true

mkdir -p "${BASE_DIR}/models" "${BASE_DIR}/results" "${BASE_DIR}/.cache"
mkdir -p "${BASE_DIR}/turn-native"

docker_args=(
  --rm
  -it
  --name "${CONTAINER_NAME}"
  --gpus all
  --env-file "${ENV_FILE}"
  -p 80:80
  -p 443:443
  -p 127.0.0.1:8780:8780
  -p 3478:3478/tcp
  -p 3478:3478/udp
  -p 5349:5349/tcp
  -p 49160-49200:49160-49200/udp
  -v "${BASE_DIR}/letsencrypt:/etc/letsencrypt"
  -v "${BASE_DIR}/models:/opt/musetalk/models"
  -v "${BASE_DIR}/results:/opt/musetalk/results"
  -v "${BASE_DIR}/.cache:/root/.cache"
  -v "${BASE_DIR}/turn-native:/opt/onebox/turn-native:ro"
)

if [[ -d "${LOCAL_MUSETALK_DIR}" ]]; then
  echo "Using local MuseTalk repo: ${LOCAL_MUSETALK_DIR}"
  docker_args+=(
    -v "${LOCAL_MUSETALK_DIR}:/opt/musetalk:ro"
  )
fi

docker run "${docker_args[@]}" "${IMAGE_TAG}"
