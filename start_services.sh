#!/usr/bin/env bash
# Modified start_services.sh - Runs the unified Python server
set -euo pipefail

log() {
  printf '[onebox] %s\n' "$*"
}

turn_pid=""
unified_server_pid=""

cleanup() {
  set +e
  if [[ -n "${unified_server_pid}" ]]; then kill "${unified_server_pid}" 2>/dev/null || true; fi
  if [[ -n "${turn_pid}" ]]; then kill "${turn_pid}" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

detect_gcp_public_ip() {
  curl -fsS -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip" 2>/dev/null || true
}

ensure_https_cert() {
  local cert_path="${ONEBOX_TLS_CERT_PATH:-/tmp/onebox.crt}"
  local key_path="${ONEBOX_TLS_KEY_PATH:-/tmp/onebox.key}"
  local cert_cn="${ONEBOX_TLS_CN:-${ONEBOX_PUBLIC_HOST:-${PUBLIC_IP:-127.0.0.1}}}"
  local cert_days="${ONEBOX_TLS_DAYS:-365}"
  local san_entry=""
  local cert_dir
  local key_dir

  cert_dir="$(dirname "${cert_path}")"
  key_dir="$(dirname "${key_path}")"
  mkdir -p "${cert_dir}" "${key_dir}"

  if [[ -s "${cert_path}" && -s "${key_path}" ]]; then
    log "using existing TLS certificate at ${cert_path}"
    return
  fi

  if [[ "${cert_cn}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    san_entry="IP:${cert_cn}"
  else
    san_entry="DNS:${cert_cn}"
  fi

  log "generating self-signed TLS certificate for CN=${cert_cn}"
  openssl req -x509 -nodes -days "${cert_days}" -newkey rsa:2048 \
    -keyout "${key_path}" \
    -out "${cert_path}" \
    -subj "/CN=${cert_cn}" \
    -addext "subjectAltName = ${san_entry}"
}

# ============================================================================
# TURN Server Configuration (No changes here)
# ============================================================================
ENABLE_TURN="${ENABLE_TURN:-1}"
TURN_PORT="${TURN_PORT:-3478}"
TURN_MIN_PORT="${TURN_MIN_PORT:-49160}"
TURN_MAX_PORT="${TURN_MAX_PORT:-49200}"
TURN_REALM="${TURN_REALM:-musetalk.local}"
TURN_BIND_IP="${TURN_BIND_IP:-0.0.0.0}"
TURN_EXTERNAL_IP="${TURN_EXTERNAL_IP:-}"
TURN_USER="${TURN_USER:-}"
TURN_PASS="${TURN_PASS:-}"
TURN_PUBLIC_HOST="${TURN_PUBLIC_HOST:-}"
TURN_ENV_FILE="${TURN_ENV_FILE:-/opt/onebox/turn-native/turn_credentials.env}"
CLOUDFLARE_TURN_TOKEN_ID="${CLOUDFLARE_TURN_TOKEN_ID:-}"
CLOUDFLARE_TURN_API_TOKEN="${CLOUDFLARE_TURN_API_TOKEN:-}"
TURN_PROVIDER="${TURN_PROVIDER:-}"

if [[ -f "${TURN_ENV_FILE}" ]]; then
  log "loading TURN credentials from ${TURN_ENV_FILE}"
  source "${TURN_ENV_FILE}"
fi

if [[ -z "${TURN_PROVIDER}" ]]; then
  if [[ -n "${CLOUDFLARE_TURN_TOKEN_ID}" && -n "${CLOUDFLARE_TURN_API_TOKEN}" ]]; then
    TURN_PROVIDER="cloudflare"
  elif [[ "${ENABLE_TURN}" == "1" ]]; then
    TURN_PROVIDER="local"
  else
    TURN_PROVIDER="off"
  fi
fi
export TURN_PROVIDER
log "TURN provider mode: ${TURN_PROVIDER}"

# ============================================================================
# Model Download Configuration
# ============================================================================
AUTO_DOWNLOAD_WEIGHTS="${AUTO_DOWNLOAD_WEIGHTS:-0}"
RESULTS_RESTORE="${RESULTS_RESTORE:-1}"

if [[ "${AUTO_DOWNLOAD_WEIGHTS}" == "1" ]]; then
  log "auto-downloading model weights and restoring avatar cache..."
  cd /opt/musetalk
  RESULTS_RESTORE="${RESULTS_RESTORE}" bash download_weights.sh
fi

# ============================================================================
# Start TURN Server (if enabled) (No changes here)
# ============================================================================
if [[ "${TURN_PROVIDER}" == "local" && "${ENABLE_TURN}" == "1" ]]; then
  if [[ -z "${TURN_USER}" || -z "${TURN_PASS}" ]]; then
    log "TURN is enabled but TURN_USER/TURN_PASS is missing. Disable TURN or set credentials."
    exit 1
  fi

  turn_private_ip="${TURN_PRIVATE_IP:-}"
  if [[ -z "${turn_private_ip}" ]]; then
    turn_private_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  fi

  turn_relay_ip="${TURN_BIND_IP}"
  if [[ "${turn_relay_ip}" == "0.0.0.0" && -n "${turn_private_ip}" ]]; then
    turn_relay_ip="${turn_private_ip}"
  fi

  log "starting coturn on ${TURN_BIND_IP}:${TURN_PORT} (relay ${TURN_MIN_PORT}-${TURN_MAX_PORT})"
  cat >/tmp/turnserver.conf <<EOF
listening-port=${TURN_PORT}
tls-listening-port=5349
listening-ip=${TURN_BIND_IP}
relay-ip=${turn_relay_ip}
fingerprint
lt-cred-mech
realm=${TURN_REALM}
user=${TURN_USER}:${TURN_PASS}
no-cli
no-multicast-peers
min-port=${TURN_MIN_PORT}
max-port=${TURN_MAX_PORT}
EOF
  if [[ -n "${TURN_EXTERNAL_IP}" ]]; then
    turn_external_ip_value="${TURN_EXTERNAL_IP}"
    if [[ "${turn_external_ip_value}" != */* && -n "${turn_private_ip}" ]]; then
      turn_external_ip_value="${turn_external_ip_value}/${turn_private_ip}"
    fi
    log "coturn external-ip mapping: ${turn_external_ip_value}"
    echo "external-ip=${turn_external_ip_value}" >>/tmp/turnserver.conf
  fi

  turnserver -c /tmp/turnserver.conf -n -v &
  turn_pid=$!
fi

# ============================================================================
# Build ICE Server Configuration (No changes here)
# ============================================================================
ice_args=()
if [[ -n "${ICE_SERVER:-}" ]]; then ice_args+=(--ice-server "${ICE_SERVER}"); fi
if [[ -n "${ICE_SERVER_2:-}" ]]; then ice_args+=(--ice-server "${ICE_SERVER_2}"); fi
if [[ -n "${ICE_SERVER_3:-}" ]]; then ice_args+=(--ice-server "${ICE_SERVER_3}"); fi

if [[ "${TURN_PROVIDER}" == "local" && "${ENABLE_TURN}" == "1" && -n "${TURN_PUBLIC_HOST}" && ${#ice_args[@]} -eq 0 ]]; then
  ice_args+=(--ice-server "turn:${TURN_PUBLIC_HOST}:${TURN_PORT}?transport=tcp")
  ice_args+=(--ice-transport-policy "relay")
  ice_args+=(--ice-username "${TURN_USER}")
  ice_args+=(--ice-credential "${TURN_PASS}")
fi

if [[ -n "${ICE_USERNAME:-}" ]]; then ice_args+=(--ice-username "${ICE_USERNAME}"); fi
if [[ -n "${ICE_CREDENTIAL:-}" ]]; then ice_args+=(--ice-credential "${ICE_CREDENTIAL}"); fi
if [[ -n "${ICE_TRANSPORT_POLICY:-}" ]]; then ice_args+=(--ice-transport-policy "${ICE_TRANSPORT_POLICY}"); fi

# ============================================================================
# Start Unified Server
# ============================================================================
# Build the MuseTalk CLI invocation and keep PersonaPlex on the same aiohttp app.
log "starting Unified Server..."
log "WebRTC port (from MUSE_PORT): ${MUSE_PORT:-8780}"
log "ICE args will be passed to the server: ${ice_args[*]:-}"

# Set environment variables for the unified server
export MUSE_HOST="${MUSE_HOST:-0.0.0.0}"
export MUSE_PORT="${MUSE_PORT:-8780}"
export MUSE_AVATAR_ID="${MUSE_AVATAR_ID:-my_avatar_720_live}"
export MUSE_VERSION="${MUSE_VERSION:-v15}"
export MUSE_GPU_ID="${MUSE_GPU_ID:-0}"
export MUSE_FPS="${MUSE_FPS:-25}"
export MUSE_BATCH_SIZE="${MUSE_BATCH_SIZE:-16}"
export MUSE_WINDOW_MS="${MUSE_WINDOW_MS:-640}"
export MUSE_HOP_MS="${MUSE_HOP_MS:-80}"
export MUSE_MIN_WINDOW_MS="${MUSE_MIN_WINDOW_MS:-320}"
export MUSE_MAX_ADVANCE_MS="${MUSE_MAX_ADVANCE_MS:-240}"
export MUSE_MAX_TAIL_FRAMES="${MUSE_MAX_TAIL_FRAMES:-5}"
export MUSE_MOUTH_SMOOTHING_ALPHA="${MUSE_MOUTH_SMOOTHING_ALPHA:-0.75}"
export MUSE_USE_FP16="${MUSE_USE_FP16:-1}"
export MUSE_REQUIRE_MMPOSE="${MUSE_REQUIRE_MMPOSE:-0}"
export MUSE_PERSONAPLEX_PATH="${MUSE_PERSONAPLEX_PATH:-/api/chat}"
export MUSE_PERSONAPLEX_TEXT_PROMPT="${MUSE_PERSONAPLEX_TEXT_PROMPT:-You enjoy having a good conversation.}"
export MUSE_PERSONAPLEX_VOICE_PROMPT="${MUSE_PERSONAPLEX_VOICE_PROMPT:-myvoice.pt}"
export MUSE_INPUT_SOURCE="${MUSE_INPUT_SOURCE:-mirror}"
export DEBUG_WEBRTC="${DEBUG_WEBRTC:-0}"
detected_public_ip=""
if [[ -z "${ONEBOX_PUBLIC_HOST:-}" ]]; then
  detected_public_ip="$(detect_gcp_public_ip)"
  if [[ -n "${detected_public_ip}" ]]; then
    export ONEBOX_PUBLIC_HOST="${detected_public_ip}"
    log "detected public host from GCP metadata: ${ONEBOX_PUBLIC_HOST}"
  fi
fi
if [[ -z "${ONEBOX_PUBLIC_HOST:-}" && -n "${TURN_PUBLIC_HOST:-}" ]]; then
  export ONEBOX_PUBLIC_HOST="${TURN_PUBLIC_HOST}"
fi
if [[ -z "${ONEBOX_PUBLIC_HOST:-}" && -n "${ONEBOX_TLS_CN:-}" ]]; then
  export ONEBOX_PUBLIC_HOST="${ONEBOX_TLS_CN}"
fi
if [[ -z "${ONEBOX_PUBLIC_HOST:-}" && -n "${PUBLIC_IP:-}" ]]; then
  export ONEBOX_PUBLIC_HOST="${PUBLIC_IP}"
fi
export ONEBOX_TLS_CERT_PATH="${ONEBOX_TLS_CERT_PATH:-/tmp/onebox.crt}"
export ONEBOX_TLS_KEY_PATH="${ONEBOX_TLS_KEY_PATH:-/tmp/onebox.key}"
export ONEBOX_TLS_ENABLE="${ONEBOX_TLS_ENABLE:-1}"
export ONEBOX_TLS_CN="${ONEBOX_TLS_CN:-${ONEBOX_PUBLIC_HOST:-127.0.0.1}}"
export ONEBOX_INTERNAL_HTTP_HOST="${ONEBOX_INTERNAL_HTTP_HOST:-127.0.0.1}"
export ONEBOX_INTERNAL_HTTP_PORT="${ONEBOX_INTERNAL_HTTP_PORT:-$((MUSE_PORT + 1))}"
export TURN_PUBLIC_HOST="${TURN_PUBLIC_HOST:-${ONEBOX_PUBLIC_HOST:-}}"
export TURN_EXTERNAL_IP="${TURN_EXTERNAL_IP:-${detected_public_ip:-}}"
default_personaplex_host="127.0.0.1"
if [[ "${MUSE_HOST}" != "0.0.0.0" && "${MUSE_HOST}" != "::" ]]; then
  default_personaplex_host="${MUSE_HOST}"
fi
export PERSONAPLEX_HOST="${PERSONAPLEX_HOST:-${ONEBOX_INTERNAL_HTTP_HOST}}" # Internal unified target host
export PERSONAPLEX_PORT="${PERSONAPLEX_PORT:-${ONEBOX_INTERNAL_HTTP_PORT}}" # Loopback HTTP listener for internal bridge

if [[ "${ONEBOX_TLS_ENABLE}" == "1" ]]; then
  ensure_https_cert
fi

unified_args=(
  --host "${MUSE_HOST}"
  --port "${MUSE_PORT}"
  --avatar-id "${MUSE_AVATAR_ID}"
  --version "${MUSE_VERSION}"
  --gpu-id "${MUSE_GPU_ID}"
  --fps "${MUSE_FPS}"
  --batch-size "${MUSE_BATCH_SIZE}"
  --window-ms "${MUSE_WINDOW_MS}"
  --hop-ms "${MUSE_HOP_MS}"
  --min-window-ms "${MUSE_MIN_WINDOW_MS}"
  --max-advance-ms "${MUSE_MAX_ADVANCE_MS}"
  --max-tail-frames "${MUSE_MAX_TAIL_FRAMES}"
  --mouth-smoothing-alpha "${MUSE_MOUTH_SMOOTHING_ALPHA}"
  --personaplex-host "${PERSONAPLEX_HOST}"
  --personaplex-port "${PERSONAPLEX_PORT}"
  --personaplex-path "${MUSE_PERSONAPLEX_PATH}"
  --personaplex-text-prompt "${MUSE_PERSONAPLEX_TEXT_PROMPT}"
  --personaplex-voice-prompt "${MUSE_PERSONAPLEX_VOICE_PROMPT}"
  --input-source "${MUSE_INPUT_SOURCE}"
)

if [[ -n "${MUSE_AVATAR_FPS:-}" ]]; then unified_args+=(--avatar-fps "${MUSE_AVATAR_FPS}"); fi
if [[ -n "${MUSE_BBOX_SHIFT:-}" ]]; then unified_args+=(--bbox-shift "${MUSE_BBOX_SHIFT}"); fi
if [[ -n "${MUSE_UNET_MODEL_PATH:-}" ]]; then unified_args+=(--unet-model-path "${MUSE_UNET_MODEL_PATH}"); fi
if [[ -n "${MUSE_UNET_CONFIG:-}" ]]; then unified_args+=(--unet-config "${MUSE_UNET_CONFIG}"); fi
if [[ -n "${MUSE_VAE_TYPE:-}" ]]; then unified_args+=(--vae-type "${MUSE_VAE_TYPE}"); fi
if [[ -n "${MUSE_WHISPER_DIR:-}" ]]; then unified_args+=(--whisper-dir "${MUSE_WHISPER_DIR}"); fi
if [[ -n "${MUSE_FFMPEG_PATH:-}" ]]; then unified_args+=(--ffmpeg-path "${MUSE_FFMPEG_PATH}"); fi
if [[ -n "${MUSE_PARSING_MODE:-}" ]]; then unified_args+=(--parsing-mode "${MUSE_PARSING_MODE}"); fi
if [[ -n "${MUSE_EXTRA_MARGIN:-}" ]]; then unified_args+=(--extra-margin "${MUSE_EXTRA_MARGIN}"); fi
if [[ -n "${MUSE_LEFT_CHEEK_WIDTH:-}" ]]; then unified_args+=(--left-cheek-width "${MUSE_LEFT_CHEEK_WIDTH}"); fi
if [[ -n "${MUSE_RIGHT_CHEEK_WIDTH:-}" ]]; then unified_args+=(--right-cheek-width "${MUSE_RIGHT_CHEEK_WIDTH}"); fi
if [[ -n "${MUSE_AUDIO_PADDING_LENGTH_LEFT:-}" ]]; then unified_args+=(--audio-padding-length-left "${MUSE_AUDIO_PADDING_LENGTH_LEFT}"); fi
if [[ -n "${MUSE_AUDIO_PADDING_LENGTH_RIGHT:-}" ]]; then unified_args+=(--audio-padding-length-right "${MUSE_AUDIO_PADDING_LENGTH_RIGHT}"); fi
if [[ -n "${MUSE_RING_BUFFER_SECONDS:-}" ]]; then unified_args+=(--ring-buffer-seconds "${MUSE_RING_BUFFER_SECONDS}"); fi
if [[ -n "${MUSE_VIDEO_QUEUE_SIZE:-}" ]]; then unified_args+=(--video-queue-size "${MUSE_VIDEO_QUEUE_SIZE}"); fi
if [[ -n "${MUSE_RECONNECT_DELAY_SECONDS:-}" ]]; then unified_args+=(--reconnect-delay-seconds "${MUSE_RECONNECT_DELAY_SECONDS}"); fi
if [[ -n "${MUSE_STATUS_JSON:-}" ]]; then unified_args+=(--status-json "${MUSE_STATUS_JSON}"); fi
if [[ -n "${MUSE_API_TOKEN:-}" ]]; then unified_args+=(--api-token "${MUSE_API_TOKEN}"); fi
if [[ -n "${MUSE_SESSION_OFFER_TIMEOUT_SECONDS:-}" ]]; then unified_args+=(--session-offer-timeout-seconds "${MUSE_SESSION_OFFER_TIMEOUT_SECONDS}"); fi
if [[ -n "${MUSE_SESSION_MAX_AGE_SECONDS:-}" ]]; then unified_args+=(--session-max-age-seconds "${MUSE_SESSION_MAX_AGE_SECONDS}"); fi
if [[ -n "${MUSE_SESSION_CLEANUP_INTERVAL_SECONDS:-}" ]]; then unified_args+=(--session-cleanup-interval-seconds "${MUSE_SESSION_CLEANUP_INTERVAL_SECONDS}"); fi
if [[ -n "${MUSE_SESSION_DISCONNECT_GRACE_SECONDS:-}" ]]; then unified_args+=(--session-disconnect-grace-seconds "${MUSE_SESSION_DISCONNECT_GRACE_SECONDS}"); fi
if [[ -n "${MUSE_PERSONAPLEX_EXTRA_QUERY:-}" ]]; then
  IFS=',' read -r -a personaplex_extra_query <<< "${MUSE_PERSONAPLEX_EXTRA_QUERY}"
  for kv in "${personaplex_extra_query[@]}"; do
    if [[ -n "${kv}" ]]; then
      unified_args+=(--personaplex-extra-query "${kv}")
    fi
  done
fi

if [[ "${MUSE_USE_FP16}" == "1" ]]; then unified_args+=(--use-fp16); fi
if [[ "${MUSE_REQUIRE_MMPOSE}" == "1" ]]; then unified_args+=(--require-mmpose); fi
if [[ "${MUSE_WEBRTC_AUDIO_LOOPBACK:-0}" == "1" ]]; then unified_args+=(--webrtc-audio-loopback); fi
if [[ "${MUSE_MUSETALK_ONLY:-0}" == "1" ]]; then unified_args+=(--musetalk-only); fi
if [[ "${MUSE_ENABLE_API_AUTH:-0}" == "1" ]]; then unified_args+=(--enable-api-auth); fi
if [[ "${MUSE_MULTI_SESSION:-0}" == "1" ]]; then unified_args+=(--multi-session); fi
if [[ "${MUSE_WEB_TEST_ONLY:-0}" == "1" ]]; then unified_args+=(--web-test-only); fi
if [[ "${DEBUG_WEBRTC}" == "1" ]]; then unified_args+=(--debug); fi
if [[ -n "${MUSE_DEBUG_EVENTS_LIMIT:-}" ]]; then unified_args+=(--debug-events-limit "${MUSE_DEBUG_EVENTS_LIMIT}"); fi

# Launch the single server process
# Pass ice_args as command-line arguments
python /opt/onebox/unified_server.py "${unified_args[@]}" "${ice_args[@]}" &
unified_server_pid=$!

log "Unified Server started (PID: ${unified_server_pid})"

# ============================================================================
# Wait for services and handle shutdown
# ============================================================================
set +e
wait -n "${unified_server_pid}" ${turn_pid:+"${turn_pid}"}
status=$?
set -e

log "a service exited (status=${status}), shutting down."
exit "${status}"
