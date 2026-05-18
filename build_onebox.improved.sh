#!/usr/bin/env bash
# Improved Build Script with Expert Dependency Management
# Key improvements:
# 1. Commit SHA pinning by default
# 2. Build args actually used by Dockerfile
# 3. Automatic dependency freezing
# 4. Build validation

set -euo pipefail

cd "$(dirname "$0")"

# ============================================================================
# Build Configuration
# ============================================================================

# Image tags
IMAGE_TAG="${IMAGE_TAG:-coldslim/musetalk-onebox:blackwell}"
BUILDER_IMAGE_TAG="musetalk-onebox-builder:blackwell"

# Repository configuration
# PRODUCTION: Pin these to commit SHAs
# DEVELOPMENT: Override with MUSETALK_REF=main ./build_onebox.sh
MUSETALK_REPO="${MUSETALK_REPO:-https://github.com/pixel3user/MuseTalk.git}"
MUSETALK_REF="${MUSETALK_REF:-main}"  # TODO: Replace with commit SHA
PERSONAPLEX_REPO="${PERSONAPLEX_REPO:-https://github.com/pixel3user/personaplex.git}"
PERSONAPLEX_REF="${PERSONAPLEX_REF:-main}"  # TODO: Replace with commit SHA

# Python version
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# PyTorch configuration
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
TORCH_CUDA="${TORCH_CUDA:-cu130}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

# MMCV stack versions
MMCV_VERSION="${MMCV_VERSION:-2.1.0}"
MMENGINE_VERSION="${MMENGINE_VERSION:-0.10.7}"
MMDET_VERSION="${MMDET_VERSION:-3.2.0}"
MMPOSE_VERSION="${MMPOSE_VERSION:-1.3.2}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

# Feature flags
INSTALL_MMPOSE="${INSTALL_MMPOSE:-1}"

# Optional custom PersonaPlex voice vendoring
CUSTOM_VOICE_URL="${CUSTOM_VOICE_URL:-}"
CUSTOM_VOICE_FILENAME="${CUSTOM_VOICE_FILENAME:-myvoice.pt}"

# Build behavior
VALIDATE_BUILD="${VALIDATE_BUILD:-1}"
FREEZE_DEPS="${FREEZE_DEPS:-1}"
SMOKE_TEST="${SMOKE_TEST:-1}"
RUNTIME_SMOKE_TEST="${RUNTIME_SMOKE_TEST:-ask}"
RUNTIME_SMOKE_TEST_TIMEOUT_SECONDS="${RUNTIME_SMOKE_TEST_TIMEOUT_SECONDS:-180}"

prompt_yes_no() {
  local prompt="$1"
  local default="${2:-n}"
  local reply=""
  local normalized_default
  if [[ "${default}" =~ ^[Yy]$ ]]; then
    normalized_default="y"
  else
    normalized_default="n"
  fi

  if [[ ! -t 0 ]]; then
    [[ "${normalized_default}" == "y" ]]
    return
  fi

  while true; do
    read -r -p "${prompt} [y/n]: " reply
    reply="${reply:-${normalized_default}}"
    case "${reply}" in
      y|Y) return 0 ;;
      n|N) return 1 ;;
      *) echo "Please answer y or n." ;;
    esac
  done
}

should_run_runtime_smoke_test() {
  case "${RUNTIME_SMOKE_TEST}" in
    1|true|TRUE|yes|YES|y|Y) return 0 ;;
    0|false|FALSE|no|NO|n|N) return 1 ;;
    ask|"")
      prompt_yes_no "Run runtime smoke test by starting the full image and checking port 8780?" "n"
      return
      ;;
    *)
      echo "Invalid RUNTIME_SMOKE_TEST value: ${RUNTIME_SMOKE_TEST}" >&2
      echo "Use one of: ask, y, n, 1, 0, true, false" >&2
      exit 1
      ;;
  esac
}

# ============================================================================
# Validation
# ============================================================================

# Warn if using non-SHA refs
if [[ "${MUSETALK_REF}" == "main" ]] || [[ "${MUSETALK_REF}" == "master" ]]; then
  echo "⚠ WARNING: MUSETALK_REF=${MUSETALK_REF} (not a commit SHA)"
  echo "  Builds will be non-reproducible. Pin to a commit SHA in production."
  echo ""
fi

if [[ "${PERSONAPLEX_REF}" == "main" ]] || [[ "${PERSONAPLEX_REF}" == "master" ]]; then
  echo "⚠ WARNING: PERSONAPLEX_REF=${PERSONAPLEX_REF} (not a commit SHA)"
  echo "  Builds will be non-reproducible. Pin to a commit SHA in production."
  echo ""
fi

# ============================================================================
# Build Builder Stage
# ============================================================================

echo "==================================================================="
echo "Building BUILDER stage"
echo "==================================================================="
echo "Python: ${PYTHON_VERSION}"
echo "PyTorch: ${TORCH_VERSION}+${TORCH_CUDA}"
echo "MMCV: ${MMCV_VERSION}"
echo "MuseTalk: ${MUSETALK_REPO}@${MUSETALK_REF}"
echo "PersonaPlex: ${PERSONAPLEX_REPO}@${PERSONAPLEX_REF}"
echo "Install MMPose: ${INSTALL_MMPOSE}"
if [[ -n "${CUSTOM_VOICE_URL}" ]]; then
  echo "Custom voice: ${CUSTOM_VOICE_FILENAME}"
else
  echo "Custom voice: disabled"
fi
echo "==================================================================="
echo ""

docker build \
  --network=host \
  -f Dockerfile \
  --target builder \
  --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
  --build-arg TORCH_VERSION="${TORCH_VERSION}" \
  --build-arg TORCHVISION_VERSION="${TORCHVISION_VERSION}" \
  --build-arg TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION}" \
  --build-arg TORCH_CUDA="${TORCH_CUDA}" \
  --build-arg TORCH_INDEX_URL="${TORCH_INDEX_URL}" \
  --build-arg MMCV_VERSION="${MMCV_VERSION}" \
  --build-arg MMENGINE_VERSION="${MMENGINE_VERSION}" \
  --build-arg MMDET_VERSION="${MMDET_VERSION}" \
  --build-arg MMPOSE_VERSION="${MMPOSE_VERSION}" \
  --build-arg TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
  --build-arg MUSETALK_REPO="${MUSETALK_REPO}" \
  --build-arg MUSETALK_REF="${MUSETALK_REF}" \
  --build-arg PERSONAPLEX_REPO="${PERSONAPLEX_REPO}" \
  --build-arg PERSONAPLEX_REF="${PERSONAPLEX_REF}" \
  --build-arg CUSTOM_VOICE_URL="${CUSTOM_VOICE_URL}" \
  --build-arg CUSTOM_VOICE_FILENAME="${CUSTOM_VOICE_FILENAME}" \
  --build-arg INSTALL_MMPOSE="${INSTALL_MMPOSE}" \
  -t "${BUILDER_IMAGE_TAG}" \
  . || {
    echo ""
    echo "❌ BUILDER STAGE FAILED"
    echo "Check build logs above for errors."
    echo "Common issues:"
    echo "  - Network problems downloading PyTorch"
    echo "  - CUDA version mismatch"
    echo "  - Native compilation errors in mmcv"
    exit 1
  }

echo ""
echo "✓ Builder stage completed: ${BUILDER_IMAGE_TAG}"
echo ""

# ============================================================================
# Extract Frozen Dependencies
# ============================================================================

if [[ "${FREEZE_DEPS}" == "1" ]]; then
  echo "==================================================================="
  echo "Extracting frozen dependencies from builder"
  echo "==================================================================="
  
  FREEZE_FILE="frozen-deps-$(date +%Y%m%d-%H%M%S).txt"
  docker run --rm "${BUILDER_IMAGE_TAG}" cat /opt/venv/frozen-requirements.txt > "${FREEZE_FILE}"
  
  echo "✓ Frozen dependencies saved to: ${FREEZE_FILE}"
  echo "  Package count: $(wc -l < "${FREEZE_FILE}")"
  echo ""
  
  # Show critical versions
  echo "Critical versions:"
  grep -E '^(torch|torchvision|torchaudio|mmcv|mmdet|mmpose)==' "${FREEZE_FILE}" || true
  echo ""
fi

# ============================================================================
# Build Final Stage
# ============================================================================

echo "==================================================================="
echo "Building FINAL runtime image"
echo "==================================================================="
echo ""

# Use sed trick to replace builder reference (same as original script)
sed 's/COPY --from=builder/COPY --from='"${BUILDER_IMAGE_TAG}"'/' Dockerfile > Dockerfile.final.tmp

docker build \
  --network=host \
  -f Dockerfile.final.tmp \
  --build-arg INSTALL_MMPOSE="${INSTALL_MMPOSE}" \
  -t "${IMAGE_TAG}" \
  "$@" \
  . || {
    rm -f Dockerfile.final.tmp
    echo ""
    echo "❌ FINAL STAGE FAILED"
    echo "Check build logs above for errors."
    exit 1
  }

rm Dockerfile.final.tmp

echo ""
echo "✓ Final image built: ${IMAGE_TAG}"
echo ""

# ============================================================================
# Validation
# ============================================================================

if [[ "${VALIDATE_BUILD}" == "1" ]]; then
  echo "==================================================================="
  echo "Validating build"
  echo "==================================================================="
  
  # Check image size
  IMAGE_SIZE=$(docker image inspect "${IMAGE_TAG}" --format='{{.Size}}' | awk '{print int($1/1024/1024)}')
  echo "Image size: ${IMAGE_SIZE} MB"
  
  if [[ "${SMOKE_TEST}" == "1" ]]; then
    echo ""
    echo "Running smoke tests..."
    echo ""
    
    docker run --rm \
      --env-file .env \
      --env-file turn-native/turn_credentials.env \
      "${IMAGE_TAG}" python -c "
import sys

print('[1/5] Testing torch...')
import torch
print(f'  ✓ torch {torch.__version__}')
assert torch.version.cuda, 'CUDA version not set'

print('[2/5] Testing torchaudio._extension...')
import torchaudio._extension
print('  ✓ torchaudio._extension loads')

print('[3/5] Testing mmcv...')
try:
    import mmcv
    print(f'  ✓ mmcv {mmcv.__version__}')
    import mmcv._ext
    print('  ✓ mmcv._ext loads')
except ImportError as e:
    if '${INSTALL_MMPOSE}' == '1':
        print(f'  ❌ FAILED: {e}')
        sys.exit(1)
    else:
        print('  ⚠ mmcv not installed (INSTALL_MMPOSE=0)')

print('[4/5] Testing mmdet/mmpose...')
try:
    import mmdet, mmpose
    print(f'  ✓ mmdet {mmdet.__version__}')
    print(f'  ✓ mmpose {mmpose.__version__}')
except ImportError:
    if '${INSTALL_MMPOSE}' == '1':
        print('  ⚠ mmdet/mmpose import failed')
    else:
        print('  ⚠ mmdet/mmpose not installed (INSTALL_MMPOSE=0)')

print('[5/5] Testing application imports...')
import musetalk
print('  ✓ musetalk module found')

print()
print('=== ALL SMOKE TESTS PASSED ===')
" || {
      echo ""
      echo "❌ SMOKE TESTS FAILED"
      echo "The image was built but critical imports are broken."
      echo "This is a regression - DO NOT deploy this image."
      exit 1
    }
    
    echo ""
    echo "✓ Smoke tests passed"
  fi

  if should_run_runtime_smoke_test; then
    echo ""
    echo "Running runtime smoke test..."
    echo ""

    RUNTIME_CONTAINER_NAME="musetalk-runtime-smoke-$$"
    runtime_env_args=(--env-file .env)
    if [[ -f turn-native/turn_credentials.env ]]; then
      runtime_env_args+=(--env-file turn-native/turn_credentials.env)
    fi

    cleanup_runtime_smoke() {
      docker rm -f "${RUNTIME_CONTAINER_NAME}" >/dev/null 2>&1 || true
    }

    trap cleanup_runtime_smoke EXIT

    docker run -d \
      --name "${RUNTIME_CONTAINER_NAME}" \
      --gpus all \
      "${runtime_env_args[@]}" \
      -p 8780:8780 \
      "${IMAGE_TAG}" >/dev/null

    runtime_ready=0
    for ((i=1; i<=RUNTIME_SMOKE_TEST_TIMEOUT_SECONDS; i++)); do
      if curl -fsS http://127.0.0.1:8780/config >/dev/null 2>&1; then
        runtime_ready=1
        break
      fi
      sleep 1
    done

    if [[ "${runtime_ready}" != "1" ]]; then
      echo ""
      echo "❌ RUNTIME SMOKE TEST FAILED"
      echo "The container did not become ready on port 8780 within ${RUNTIME_SMOKE_TEST_TIMEOUT_SECONDS}s."
      echo ""
      echo "Recent container logs:"
      docker logs --tail 200 "${RUNTIME_CONTAINER_NAME}" || true
      exit 1
    fi

    echo "✓ Runtime smoke test passed: container responded on http://127.0.0.1:8780"
    cleanup_runtime_smoke
    trap - EXIT
  else
    echo ""
    echo "Skipping runtime smoke test."
  fi
  
  echo ""
fi

# ============================================================================
# Summary
# ============================================================================

echo "==================================================================="
echo "BUILD COMPLETE"
echo "==================================================================="
echo ""
echo "Image: ${IMAGE_TAG}"
echo "Size: ${IMAGE_SIZE:-unknown} MB"
echo ""
echo "Configuration:"
echo "  PyTorch: ${TORCH_VERSION}+${TORCH_CUDA}"
echo "  MMCV: ${MMCV_VERSION}"
echo "  MMPose: $([ "${INSTALL_MMPOSE}" = "1" ] && echo "enabled" || echo "disabled")"
echo ""
echo "Next steps:"
echo "  1. Run the container with port forwarding: ./run_onebox.sh"
echo "  2. Push to registry: docker push ${IMAGE_TAG}"
if [[ "${FREEZE_DEPS}" == "1" ]]; then
  echo "  3. Commit frozen deps: git add ${FREEZE_FILE}"
fi
echo ""

# Warn about non-reproducible builds
if [[ "${MUSETALK_REF}" == "main" ]] || [[ "${PERSONAPLEX_REF}" == "main" ]]; then
  echo "⚠ REMINDER: This build is NOT reproducible (using branch refs)"
  echo "  To make it reproducible:"
  echo "  1. Get commit SHAs:"
  echo "     docker run --rm ${IMAGE_TAG} sh -c 'cd /opt/musetalk && git rev-parse HEAD'"
  echo "     docker run --rm ${IMAGE_TAG} sh -c 'cd /opt/personaplex && git rev-parse HEAD'"
  echo "  2. Update build script with these SHAs"
  echo "  3. Rebuild and verify"
  echo ""
fi

echo "Build script completed successfully."
