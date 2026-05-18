# Fixed Multi-Stage Dockerfile with Expert Dependency Management
# Key improvements:
# 1. Single venv built in builder, copied to runtime (no pip in runtime)
# 2. Actual use of build args (not hardcoded versions)
# 3. Protected package installation
# 4. Build-time smoke tests
# 5. Model artifact validation

# ============================================================================
# Stage 1: Builder - Build complete Python environment ONCE
# ============================================================================
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Build configuration - these actually get USED now
ARG PYTHON_VERSION=3.10
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG TORCH_CUDA=cu130
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

# MMCV stack versions
ARG MMCV_VERSION=2.1.0
ARG MMENGINE_VERSION=0.10.7
ARG MMDET_VERSION=3.2.0
ARG MMPOSE_VERSION=1.3.2
ARG TORCH_CUDA_ARCH_LIST="9.0"

# Repository configuration - SHOULD BE COMMIT SHAs IN PRODUCTION
ARG MUSETALK_REPO=https://github.com/pixel3user/MuseTalk.git
ARG MUSETALK_REF=main
ARG PERSONAPLEX_REPO=https://github.com/pixel3user/personaplex.git
ARG PERSONAPLEX_REF=main
ARG CUSTOM_VOICE_URL=""
ARG CUSTOM_VOICE_FILENAME="myvoice.pt"

# Feature flags
ARG INSTALL_MMPOSE=1

# Install build dependencies
RUN rm -rf /var/lib/apt/lists/* && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    ffmpeg \
    curl \
    ca-certificates \
    build-essential \
    ninja-build \
    pkg-config \
    libgirepository1.0-dev \
 && rm -rf /var/lib/apt/lists/*

# Create virtual environment (this is the ONLY Python environment)
RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in venv
RUN pip install --no-cache-dir --upgrade pip "setuptools<70" wheel

# ============================================================================
# Install PyTorch ONCE - using ARG variables this time
# ============================================================================
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION}+${TORCH_CUDA} \
    torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA} \
    torchaudio==${TORCHAUDIO_VERSION}+${TORCH_CUDA} \
    --index-url ${TORCH_INDEX_URL}

# ============================================================================
# Build mmcv/mmdet/mmpose from source (if enabled)
# ============================================================================
WORKDIR /wheels

RUN if [ "${INSTALL_MMPOSE}" = "1" ]; then \
      pip install --no-cache-dir "mmengine==${MMENGINE_VERSION}" && \
      MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        pip wheel --no-build-isolation "mmcv==${MMCV_VERSION}" && \
      pip wheel --no-build-isolation "chumpy==0.70" && \
      pip wheel --no-build-isolation "mmdet==${MMDET_VERSION}" && \
      pip wheel --no-build-isolation "mmpose==${MMPOSE_VERSION}" && \
      # Install wheels directly into venv (not separate stage)
      pip install --no-cache-dir --no-deps --find-links=/wheels /wheels/*.whl; \
    else \
      echo "INSTALL_MMPOSE=0 — skipping mmcv/mmdet/mmpose."; \
    fi

# ============================================================================
# Clone application repositories
# ============================================================================
WORKDIR /opt
RUN git clone --depth 1 --branch "${MUSETALK_REF}" "${MUSETALK_REPO}" /opt/musetalk
RUN git clone --depth 1 --branch "${PERSONAPLEX_REF}" "${PERSONAPLEX_REPO}" /opt/personaplex
RUN sed -i \
      -e 's/"torch>=2.2.0,<2.5"/"torch>=2.2.0"/' \
      -e 's/"huggingface-hub>=0.24,<0.25"/"huggingface-hub>=0.24"/' \
      -e 's/"sounddevice==0.5"/"sounddevice>=0.5"/' \
      /opt/personaplex/moshi/pyproject.toml

# Fix MuseTalk download script
RUN sed -i 's/pip install -U "huggingface_hub\[cli\]"/pip install "huggingface_hub>=0.24,<0.25"/g' /opt/musetalk/download_weights.sh && \
    sed -i 's/\<huggingface-cli\>/hf/g' /opt/musetalk/download_weights.sh

# Create hf wrapper script
RUN printf '#!/bin/sh\npython -m huggingface_hub.commands.huggingface_cli "$@"\n' > /usr/bin/hf && \
    chmod +x /usr/bin/hf

# ============================================================================
# Install application requirements WITH PROTECTION
# ============================================================================

# Create protected package list (these must NOT be replaced)
RUN cat > /tmp/protected-packages.txt <<'EOF'
torch
torchvision
torchaudio
mmcv
mmcv-lite
mmengine
mmdet
mmpose
openmim
EOF

# Install MuseTalk requirements, excluding protected packages
RUN pip install "opencv-python-headless<4.9" && \
    grep -v '^#' /opt/musetalk/requirements.txt | \
    grep -vFf /tmp/protected-packages.txt | \
    pip install --no-cache-dir -r /dev/stdin

# Install runtime dependencies for PersonaPlex and WebRTC
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    aiohttp==3.10.11 \
    aiortc==1.14.0 \
    av==16.0.1 \
    sentencepiece==0.2.0 \
    sphn==0.1.12 \
    safetensors==0.4.5 \
    sounddevice==0.5.2 \
    einops==0.7.0 \
    hf_xet==1.1.10

# Install PersonaPlex package (no-deps to prevent conflicts)
RUN pip install --no-cache-dir -e /opt/personaplex/moshi --no-deps

# ============================================================================
# Vendor custom PersonaPlex voices (optional)
# ============================================================================
RUN mkdir -p /opt/personaplex/custom_voices && \
    if [ -n "${CUSTOM_VOICE_URL}" ]; then \
      echo "Downloading custom PersonaPlex voice to /opt/personaplex/custom_voices/${CUSTOM_VOICE_FILENAME}" && \
      curl -L --fail --retry 3 --retry-delay 2 \
        "${CUSTOM_VOICE_URL}" \
        -o "/opt/personaplex/custom_voices/${CUSTOM_VOICE_FILENAME}" && \
      test -s "/opt/personaplex/custom_voices/${CUSTOM_VOICE_FILENAME}"; \
    else \
      echo "No CUSTOM_VOICE_URL provided; skipping custom voice download."; \
    fi

# ============================================================================
# Vendor runtime downloads (eliminate network dependencies)
# ============================================================================

# Pre-download Silero VAD model
RUN python -c "import torch; print('Pre-downloading Silero VAD...'); model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', source='github', trust_repo=True); print('✓ Silero VAD cached')" && \
    cp -r /root/.cache/torch/hub /opt/torch-hub-cache

# ============================================================================
# Freeze final dependency state
# ============================================================================
RUN pip freeze > /opt/venv/frozen-requirements.txt && \
    echo "=== Frozen Python Environment ===" && \
    cat /opt/venv/frozen-requirements.txt

# SMOKE TESTS - Fail build if anything is broken
# ============================================================================
COPY smoke_test.py /opt/
RUN echo "=== Running Build-Time Smoke Tests ===" && \
    python /opt/smoke_test.py

# ============================================================================
# Stage 2: Runtime - ONLY copy artifacts, NO pip install
# ============================================================================
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_MMPOSE=1

# Install ONLY runtime system dependencies (no Python build tools)
RUN rm -rf /var/lib/apt/lists/* && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-venv \
    git \
    ffmpeg \
    curl \
    openssl \
    ca-certificates \
    build-essential \
    libopus0 \
    libportaudio2 \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    coturn \
    netcat-openbsd \
 && rm -rf /var/lib/apt/lists/*

# Copy COMPLETE virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=builder /opt/musetalk /opt/musetalk
COPY --from=builder /opt/personaplex /opt/personaplex

# Copy pre-downloaded models/caches
COPY --from=builder /opt/torch-hub-cache /root/.cache/torch/hub

# Copy hf wrapper
COPY --from=builder /usr/bin/hf /usr/bin/hf

# Copy frozen requirements for inspection
COPY --from=builder /opt/venv/frozen-requirements.txt /opt/

# Copy startup scripts
COPY start_services.sh /opt/onebox/start_services.sh
COPY unified_server.py /opt/onebox/unified_server.py
RUN chmod +x /opt/onebox/start_services.sh

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONPATH=/opt/personaplex/moshi:/opt/musetalk:/opt/personaplex:/opt/personaplex
ENV TORCH_HOME=/root/.cache/torch
ENV VOICE_PROMPT_DIR=/opt/personaplex/custom_voices

# Validate runtime environment (quick sanity check)
RUN python -c "import torch; import mmcv; print(f'Runtime check: torch={torch.__version__}, mmcv={mmcv.__version__}')"

# Expose ports
EXPOSE 8780 8998 3478/tcp 3478/udp 5349/tcp

ENTRYPOINT ["/opt/onebox/start_services.sh"]
