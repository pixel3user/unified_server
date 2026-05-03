#!/usr/bin/env bash
# Dependency Diff and Inspection Tool
# Usage: ./inspect-dependencies.sh <command> [args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Dependency Diff and Inspection Tool

COMMANDS:
  freeze <image>              Extract frozen requirements from image
  diff <image1> <image2>      Compare Python packages between images
  native <image>              Show native library versions
  torch <image>               Show PyTorch/CUDA configuration
  smoke <image>               Run smoke tests in image
  audit <image>               Full dependency audit

EXAMPLES:
  # Extract current dependency state
  ./inspect-dependencies.sh freeze musetalk-onebox:latest > frozen-$(date +%Y%m%d).txt
  
  # Compare before/after rebuild
  ./inspect-dependencies.sh diff musetalk-onebox:old musetalk-onebox:new
  
  # Check torch configuration
  ./inspect-dependencies.sh torch musetalk-onebox:latest
  
  # Full audit report
  ./inspect-dependencies.sh audit musetalk-onebox:latest > audit-report.txt

EOF
  exit 1
}

# ============================================================================
# Command: freeze - Extract frozen requirements
# ============================================================================
cmd_freeze() {
  local image="$1"
  echo "# Frozen requirements from ${image}"
  echo "# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""
  docker run --rm "${image}" pip freeze
}

# ============================================================================
# Command: diff - Compare Python packages
# ============================================================================
cmd_diff() {
  local image1="$1"
  local image2="$2"
  
  echo "=== Python Package Diff ==="
  echo "Comparing: ${image1} → ${image2}"
  echo ""
  
  local tmp1=$(mktemp)
  local tmp2=$(mktemp)
  trap "rm -f ${tmp1} ${tmp2}" EXIT
  
  docker run --rm "${image1}" pip freeze | sort > "${tmp1}"
  docker run --rm "${image2}" pip freeze | sort > "${tmp2}"
  
  if diff -u "${tmp1}" "${tmp2}"; then
    echo "✓ No differences found"
  fi
  
  echo ""
  echo "=== Package Count ==="
  echo "${image1}: $(wc -l < "${tmp1}") packages"
  echo "${image2}: $(wc -l < "${tmp2}") packages"
}

# ============================================================================
# Command: native - Show native library versions
# ============================================================================
cmd_native() {
  local image="$1"
  
  echo "=== Native Libraries ==="
  echo "Image: ${image}"
  echo ""
  
  echo "--- CUDA Libraries ---"
  docker run --rm "${image}" bash -c 'ldconfig -p | grep -E "cuda|cudnn|cublas|cufft" | sort'
  
  echo ""
  echo "--- Python Extensions ---"
  docker run --rm "${image}" bash -c '
    python -c "
import os, glob

ext_dirs = [
    \"/opt/venv/lib/python*/site-packages/mmcv\",
    \"/opt/venv/lib/python*/site-packages/torch\",
    \"/opt/venv/lib/python*/site-packages/torchaudio\"
]

for pattern in ext_dirs:
    for base_dir in glob.glob(pattern):
        if not os.path.exists(base_dir):
            continue
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f.endswith(\".so\"):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, base_dir)
                    size = os.path.getsize(full_path)
                    print(f\"{os.path.basename(base_dir)}/{rel_path}: {size:,} bytes\")
"
  '
}

# ============================================================================
# Command: torch - Show PyTorch configuration
# ============================================================================
cmd_torch() {
  local image="$1"
  
  echo "=== PyTorch Configuration ==="
  echo "Image: ${image}"
  echo ""
  
  docker run --rm "${image}" python -c "
import torch

print('PyTorch Version:', torch.__version__)
print('CUDA Version:', torch.version.cuda)
print('cuDNN Version:', torch.backends.cudnn.version())
print('CUDA Available:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('CUDA Device Count:', torch.cuda.device_count())
    print('CUDA Device Name:', torch.cuda.get_device_name(0))
    print('CUDA Capability:', torch.cuda.get_device_capability(0))

print()
print('Build Configuration:')
print(torch.__config__.show())
"
}

# ============================================================================
# Command: smoke - Run smoke tests
# ============================================================================
cmd_smoke() {
  local image="$1"
  
  echo "=== Smoke Tests ==="
  echo "Image: ${image}"
  echo ""
  
  docker run --rm "${image}" python -c "
import sys

print('[1/6] Testing torch...')
import torch
print(f'  ✓ torch {torch.__version__}')

print('[2/6] Testing torchvision...')
import torchvision
print(f'  ✓ torchvision {torchvision.__version__}')

print('[3/6] Testing torchaudio...')
import torchaudio
print(f'  ✓ torchaudio {torchaudio.__version__}')

print('[4/6] Testing torchaudio._extension...')
import torchaudio._extension
print('  ✓ torchaudio._extension loads')

print('[5/6] Testing mmcv...')
try:
    import mmcv
    print(f'  ✓ mmcv {mmcv.__version__}')
    
    import mmcv._ext
    print('  ✓ mmcv._ext loads (CUDA ops available)')
except ImportError as e:
    print(f'  ⚠ mmcv unavailable: {e}')

print('[6/6] Testing mmdet/mmpose...')
try:
    import mmdet, mmpose
    print(f'  ✓ mmdet {mmdet.__version__}')
    print(f'  ✓ mmpose {mmpose.__version__}')
except ImportError:
    print('  ⚠ mmdet/mmpose not installed')

print()
print('=== Dependency Check ===')
sys.exit(0)  # Skip pip check in this example
"
  
  echo ""
  echo "=== pip check ==="
  docker run --rm "${image}" pip check || echo "⚠ Dependency conflicts found"
}

# ============================================================================
# Command: audit - Full dependency audit
# ============================================================================
cmd_audit() {
  local image="$1"
  
  echo "============================================================"
  echo "FULL DEPENDENCY AUDIT"
  echo "Image: ${image}"
  echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "============================================================"
  echo ""
  
  cmd_torch "${image}"
  echo ""
  echo "============================================================"
  cmd_smoke "${image}"
  echo ""
  echo "============================================================"
  cmd_native "${image}"
  echo ""
  echo "============================================================"
  echo "=== Full Package List ==="
  cmd_freeze "${image}"
}

# ============================================================================
# Main dispatch
# ============================================================================
if [[ $# -lt 1 ]]; then
  usage
fi

command="$1"
shift

case "${command}" in
  freeze)
    [[ $# -ne 1 ]] && usage
    cmd_freeze "$@"
    ;;
  diff)
    [[ $# -ne 2 ]] && usage
    cmd_diff "$@"
    ;;
  native)
    [[ $# -ne 1 ]] && usage
    cmd_native "$@"
    ;;
  torch)
    [[ $# -ne 1 ]] && usage
    cmd_torch "$@"
    ;;
  smoke)
    [[ $# -ne 1 ]] && usage
    cmd_smoke "$@"
    ;;
  audit)
    [[ $# -ne 1 ]] && usage
    cmd_audit "$@"
    ;;
  *)
    echo "Unknown command: ${command}"
    usage
    ;;
esac
