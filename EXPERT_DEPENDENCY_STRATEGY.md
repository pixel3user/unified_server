# Expert Container Dependency Strategy
## Root Cause Analysis

### What Your Logs Actually Show

```
OSError: libtorchaudio.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

**This is NOT a missing dependency.** This is an **ABI mismatch** — your native extensions (mmcv, torchaudio) were compiled against one PyTorch build and are now being loaded into a different one.

### The Three Failure Modes in Your Stack

1. **Binary ABI Mismatches** (your current blocker)
   - mmcv/_ext.so compiled against Torch 2.9.0+cu130
   - libtorchaudio.so linked to different CUDA symbols
   - Happens when: pip install operations run AFTER native builds

2. **Runtime Network Dependencies** (future production outage)
   ```python
   torch.hub.load('snakers4/silero-vad', 'silero_vad', source='github', branch='master')
   ```
   - Pulling from GitHub master at container startup
   - Missing models/sd-vae artifact
   - Will fail when: GitHub is down, rate-limited, or repo changes

3. **Non-deterministic Builds** (reproducibility failure)
   - `MUSETALK_REF=main` and `PERSONAPLEX_REF=main`
   - Moving HEAD means different commits on different build days
   - Your "working" image today won't build tomorrow

---

## Your Proposed Solutions: Assessment

### ✅ CORRECT Solutions

1. **"Build Python environment once, in one place"** — YES
   - Current problem: 5 separate `pip install` stages in final image
   - Lines 99-103, 107-111, 125-127, 130-140, 143 all mutate the environment

2. **"Lock the dependency graph"** — YES
   - But you need MORE than just requirements.txt
   - Need: commit SHAs, binary fingerprints, model checksums

3. **"Separate fragile extras from core runtime"** — YES
   - MMPose is 90% of your build pain
   - Most users don't need it (you have MUSE_REQUIRE_MMPOSE=0)

4. **"Remove runtime downloads"** — YES
   - Silero VAD must be vendored or pre-downloaded
   - sd-vae model must be in the image

5. **"Add build-time smoke tests"** — YES
   - Critical: test `import mmcv._ext` DURING build
   - Current setup ships broken images

6. **"Stop letting upstream requirements overwrite critical packages"** — YES
   - But your filter is incomplete (only excludes torch/torchvision/torchaudio)

### ❌ MISSING Solutions

7. **Fix your fake build args**
   - `build_onebox.sh` exposes `TORCH_VERSION`, `TORCH_INDEX_URL`, etc.
   - `Dockerfile.multistage` IGNORES these and hardcodes torch==2.9.0+cu130
   - This creates false confidence in reproducibility

8. **Fix runtime variable override hell**
   - `start_services.sh` lines 150-162 hardcode:
     ```bash
     FPS=25 BATCH_SIZE=8 WINDOW_MS=640 HOP_MS=160 ...
     ```
   - Completely ignores MUSE_FPS, MUSE_BATCH_SIZE, etc. declared earlier
   - User changes MUSE_FPS=30 → nothing happens

9. **Eliminate multi-stage pip installs in runtime image**
   - Current: builder creates wheels → runtime re-installs torch → installs wheels → installs more packages
   - Correct: builder creates complete venv → runtime ONLY copies venv

10. **Add dependency inspection and freezing**
    - You need `pip freeze` at end of builder
    - You need checksums of all .whl files
    - You need ability to audit "what changed" between builds

---

## The Expert Playbook: Step-by-Step

### Phase 1: Stop the Bleeding (Fix ABI Mismatches)

#### Change 1: Single-Stage Python Environment

**Problem:** Multiple pip install stages in final image allow version conflicts.

**Solution:** Build complete venv in builder, copy to runtime.

```dockerfile
# Stage 1: Builder
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04 AS builder

ARG PYTHON_VERSION=3.10
ARG TORCH_VERSION=2.9.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130

# Install Python and create venv
RUN apt-get update && apt-get install -y python${PYTHON_VERSION} python3-venv && \
    python${PYTHON_VERSION} -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# Install base stack ONCE
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        torch==${TORCH_VERSION}+cu130 \
        torchvision==0.24.0+cu130 \
        torchaudio==${TORCH_VERSION}+cu130 \
        --index-url ${TORCH_INDEX}

# Build native wheels
WORKDIR /wheels
RUN pip install --no-cache-dir mmengine==0.10.7 && \
    MMCV_WITH_OPS=1 FORCE_CUDA=1 pip wheel --no-build-isolation mmcv==2.1.0 && \
    pip wheel --no-build-isolation chumpy==0.70 mmdet==3.2.0 mmpose==1.3.2

# Install wheels into venv (NOT separate)
RUN pip install --no-cache-dir --no-deps --find-links=/wheels /wheels/*.whl

# Install application requirements ONCE
COPY requirements-frozen.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-frozen.txt

# Freeze final state
RUN pip freeze > /opt/venv/installed.txt

# Stage 2: Runtime
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy COMPLETE venv
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ZERO pip install operations after this point
```

**Why this works:**
- All Python packages installed in builder venv
- Native extensions compiled against final torch version
- Runtime image only receives frozen venv
- No opportunity for version conflicts

#### Change 2: Protect mmcv/mmdet/mmpose from Dependency Hell

**Problem:** Installing musetalk requirements can pull in newer torch.

**Current broken filter:**
```bash
grep -viE '^torch(vision|audio)?[>=<! ]|^torch$'
```

**Fixed filter:**
```bash
# Create exclusion list
cat > /tmp/protected-packages.txt <<EOF
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

# Install with protection
grep -v '^#' /opt/musetalk/requirements.txt | \
  grep -vFf /tmp/protected-packages.txt | \
  pip install --no-cache-dir -r /dev/stdin
```

#### Change 3: Make Build Args Actually Work

**Problem:** Dockerfile hardcodes versions instead of using ARG.

**Fix:**
```dockerfile
ARG TORCH_VERSION=2.9.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130

# USE the args
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION}+cu130 \
    torchvision==0.24.0+cu130 \
    torchaudio==${TORCH_VERSION}+cu130 \
    --index-url ${TORCH_INDEX}
```

### Phase 2: Lock Everything Down

#### Change 4: Pin to Commit SHAs

**Problem:** `MUSETALK_REF=main` means different code every build.

**Solution:**
```bash
# In build_onebox.sh
MUSETALK_REF="${MUSETALK_REF:-a1b2c3d4e5f6...}"  # actual commit SHA
PERSONAPLEX_REF="${PERSONAPLEX_REF:-f6e5d4c3b2a1...}"

# For local development, allow override:
# MUSETALK_REF=main ./build_onebox.sh
```

**Workflow:**
1. Test with `main` branch
2. Once working, get commit SHA: `git rev-parse HEAD`
3. Update default in build script
4. Re-test with pinned SHA

#### Change 5: Create Frozen Requirements

**Generate constraints file:**
```bash
# In builder stage, after all installs:
pip freeze > /opt/frozen-constraints.txt

# Include in runtime stage:
COPY --from=builder /opt/frozen-constraints.txt /opt/

# For future rebuilds, use as constraint:
pip install -r requirements.txt -c /opt/frozen-constraints.txt
```

**Store in repo:**
```bash
docker run --rm musetalk-onebox-builder cat /opt/frozen-constraints.txt \
  > constraints-$(date +%Y%m%d).txt
git add constraints-*.txt
```

#### Change 6: Vendor Runtime Downloads

**Problem:** `torch.hub.load(..., source='github', branch='master')`

**Solution A: Pre-download in builder**
```dockerfile
# In builder stage
RUN python -c "
import torch
torch.hub.load('snakers4/silero-vad', 'silero_vad', source='github', trust_repo=True)
" && \
    cp -r /root/.cache/torch/hub /opt/torch-hub-cache

# In runtime stage
COPY --from=builder /opt/torch-hub-cache /root/.cache/torch/hub
ENV TORCH_HOME=/root/.cache/torch
```

**Solution B: Vendor into repo**
```bash
# Outside Docker
git clone https://github.com/snakers4/silero-vad.git vendor/silero-vad
cd vendor/silero-vad && git rev-parse HEAD > COMMIT_SHA

# In Dockerfile
COPY vendor/silero-vad /opt/vendor/silero-vad

# In code, replace torch.hub.load with local import
```

#### Change 7: Validate Model Artifacts

**Problem:** Missing `models/sd-vae` fails at runtime.

**Fix: Fail fast during build**
```dockerfile
# After downloading weights
RUN python -c "
import os
required_models = [
    'models/sd-vae/diffusion_pytorch_model.safetensors',
    'models/musetalkV15/unet.pth',
    'models/dwpose/dw-ll_ucoco_384.pth',
]
missing = [m for m in required_models if not os.path.exists(m)]
if missing:
    raise FileNotFoundError(f'Missing models: {missing}')
print('✓ All required models present')
"
```

### Phase 3: Add Safety Nets

#### Change 8: Build-Time Smoke Tests

**Add before final stage completes:**
```dockerfile
# Test native imports
RUN python -c "
import sys
import torch, torchvision, torchaudio
print(f'torch: {torch.__version__}, cuda: {torch.version.cuda}')

import torchaudio._extension
print('✓ torchaudio._extension loads')

import mmcv
print(f'mmcv: {mmcv.__version__}')

import mmcv._ext
print('✓ mmcv._ext loads (CUDA ops available)')

import mmdet, mmpose
print(f'mmdet: {mmdet.__version__}, mmpose: {mmpose.__version__}')

print('=== ALL CRITICAL IMPORTS SUCCESSFUL ===')
"

# Validate no broken dependencies
RUN pip check || (echo "DEPENDENCY CONFLICTS DETECTED" && exit 1)
```

**This test will catch:**
- ABI mismatches (mmcv._ext fails to load)
- Missing dependencies (pip check fails)
- Version conflicts

**If this passes in builder → runtime will work**

#### Change 9: Fix Runtime Variable Override

**Problem:** `start_services.sh` line 150 overrides all MUSE_* variables.

**Fix:**
```bash
# Replace hardcoded block with:
log "starting MuseTalk WebRTC on ${MUSE_HOST}:${MUSE_PORT}"
cd /opt/musetalk

env \
  FPS="${MUSE_FPS}" \
  BATCH_SIZE="${MUSE_BATCH_SIZE}" \
  WINDOW_MS="${MUSE_WINDOW_MS}" \
  HOP_MS="${MUSE_HOP_MS}" \
  MIN_WINDOW_MS="${MUSE_MIN_WINDOW_MS}" \
  MAX_ADVANCE_MS="${MUSE_MAX_ADVANCE_MS}" \
  MAX_TAIL_FRAMES="${MUSE_MAX_TAIL_FRAMES}" \
  PERSONAPLEX_HOST="127.0.0.1" \
  PERSONAPLEX_PORT="${PERSONAPLEX_PORT}" \
  PERSONAPLEX_PATH="${MUSE_PERSONAPLEX_PATH}" \
  INPUT_SOURCE=mirror \
  "${ice_args[@]}" \
  ./scripts/run_my_avatar_720_live_webrtc.sh --avatar-fps="${MUSE_FPS}"
```

**Or better: use a config file**
```bash
cat > /tmp/muse-runtime.env <<EOF
FPS=${MUSE_FPS}
BATCH_SIZE=${MUSE_BATCH_SIZE}
WINDOW_MS=${MUSE_WINDOW_MS}
...
EOF

set -a; source /tmp/muse-runtime.env; set +a
./scripts/run_my_avatar_720_live_webrtc.sh
```

#### Change 10: Add Dependency Diff Tool

**Create inspection script:**
```bash
#!/usr/bin/env bash
# scripts/diff-dependencies.sh

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <image1> <image2>"
  exit 1
fi

echo "=== Python package diff ==="
diff -u \
  <(docker run --rm "$1" pip freeze | sort) \
  <(docker run --rm "$2" pip freeze | sort)

echo -e "\n=== Native library diff ==="
diff -u \
  <(docker run --rm "$1" ldconfig -p | grep cuda | sort) \
  <(docker run --rm "$2" ldconfig -p | grep cuda | sort)
```

**Usage:**
```bash
# Compare before/after rebuild
./scripts/diff-dependencies.sh \
  musetalk-onebox:old \
  musetalk-onebox:new
```

---

## Practical Implementation Roadmap

### Week 1: Stop the Immediate Pain

**Day 1-2: Emergency Fix**
- [ ] Apply Change 1 (single venv)
- [ ] Apply Change 2 (protected packages filter)
- [ ] Apply Change 8 (build smoke tests)
- **Goal:** Get one clean build that passes tests

**Day 3: Validate Fix**
- [ ] Run container
- [ ] Verify mmcv._ext loads: `docker exec <container> python -c "import mmcv._ext; print('OK')"`
- [ ] Verify silero-vad loads (may still download, that's OK for now)
- **Goal:** Confirm ABI mismatch is fixed

**Day 4-5: Lock Current State**
- [ ] Apply Change 4 (pin commit SHAs)
- [ ] Apply Change 5 (frozen constraints)
- [ ] Test rebuild from frozen state
- **Goal:** Reproducible builds

### Week 2: Eliminate Runtime Dependencies

**Day 1-2: Vendor Dependencies**
- [ ] Apply Change 6 (vendor silero-vad)
- [ ] Apply Change 7 (validate model artifacts)
- **Goal:** Hermetic container (no network at runtime)

**Day 3-4: Production Hardening**
- [ ] Apply Change 9 (fix runtime vars)
- [ ] Add health checks
- [ ] Add logging
- **Goal:** Production-ready container

**Day 5: Documentation**
- [ ] Document all build args
- [ ] Document all runtime envs
- [ ] Create troubleshooting guide
- **Goal:** Team can maintain this

### Week 3: Advanced Improvements

**Day 1-2: Multi-Image Strategy**
- [ ] Split into `onebox-core` and `onebox-pose`
- [ ] Test fallback path (no mmpose)
- **Goal:** Faster builds, optional features

**Day 2-3: CI/CD Integration**
- [ ] Add GitHub Actions / GitLab CI
- [ ] Automate constraint generation
- [ ] Automate smoke tests
- **Goal:** Catch regressions early

**Day 4-5: Monitoring & Observability**
- [ ] Add dependency diff tool (Change 10)
- [ ] Track build times
- [ ] Track image sizes
- **Goal:** Continuous improvement

---

## Quick Wins You Can Do TODAY

### 1. Add smoke tests to existing Dockerfile

Add this at the very end of `Dockerfile.multistage`:

```dockerfile
# === SMOKE TESTS ===
RUN python -c "import mmcv._ext; print('✓ mmcv native ops OK')" || \
    (echo "FATAL: mmcv._ext failed to load" && exit 1)

RUN python -c "import torchaudio._extension; print('✓ torchaudio OK')" || \
    (echo "FATAL: torchaudio extension failed" && exit 1)

RUN pip check || (echo "FATAL: dependency conflicts" && exit 1)
```

**This will make your build FAIL instead of producing broken images.**

### 2. Pin your repos TODAY

In `build_onebox.sh`, change:
```bash
MUSETALK_REF="${MUSETALK_REF:-main}"
```

To:
```bash
# Get current HEAD: cd musetalk-clone && git rev-parse HEAD
MUSETALK_REF="${MUSETALK_REF:-a1b2c3d4e5f67890abcdef...}"
```

**Document the commit somewhere:**
```bash
# Pin to working commit as of 2024-XX-XX
# Tested with: <commit message>
# Last known good build: ...
```

### 3. Protect mmcv from being replaced

In your Dockerfile, change:
```dockerfile
RUN grep -v '^#' /opt/musetalk/requirements.txt \
    | grep -viE '^torch(vision|audio)?[>=<! ]|^torch$' \
    | pip install --no-cache-dir -r /dev/stdin
```

To:
```dockerfile
RUN grep -v '^#' /opt/musetalk/requirements.txt \
    | grep -viE '^(torch|mmcv|mmdet|mmpose|mmengine|openmim)' \
    | pip install --no-cache-dir -r /dev/stdin
```

---

## Common Pitfall: The "It Works On My Machine" Trap

**Scenario:** You rebuild today, it works. You rebuild tomorrow with SAME CODE, it breaks.

**Why:**
- MuseTalk `main` branch got a new commit
- New commit added a dependency
- That dependency has a newer transitive dep for torch
- Newer torch overwrites your mmcv-compatible version
- mmcv._ext compiled against old torch, loaded into new torch
- ABI mismatch → crash

**Prevention:**
- Pin commits (Change 4)
- Freeze constraints (Change 5)
- Smoke tests (Change 8)

All three together = reproducible builds forever.

---

## Decision Matrix: What to Do First

| Problem | Impact | Effort | Priority | Quick Win? |
|---------|--------|--------|----------|------------|
| ABI mismatch | BLOCKER | Medium | **P0** | Smoke tests (2 min) |
| Fake build args | Confusion | Low | P2 | No |
| Runtime downloads | Fragility | Medium | P1 | No |
| Moving targets | Reproducibility | Low | **P0** | Pin SHAs (5 min) |
| Multi-stage pip | Root cause | High | **P0** | No |
| Runtime var override | UX bug | Low | P2 | No |
| Missing validation | Silent failures | Low | **P0** | Smoke tests (2 min) |

**Do P0 items first. They're 90% of the pain.**

---

## What Success Looks Like

### Before (Current State)
```
$ docker build -t test .
... 30 minutes later ...
Successfully built abc123

$ docker run test
[crash with undefined symbol]
```

### After (Expert State)
```
$ docker build -t test .
... 25 minutes later ...
Running smoke tests...
✓ mmcv native ops OK
✓ torchaudio OK
✓ pip check passed
✓ All required models present
Successfully built abc123

$ docker run test
[runs perfectly, no network calls, all deps present]

$ docker run test  # 6 months later, still works
```

**Your builds become:**
- **Deterministic** — same input = same output
- **Hermetic** — no runtime network dependencies
- **Validated** — broken images never leave builder
- **Debuggable** — frozen constraints + diff tool
- **Maintainable** — clear separation of concerns

---

## The Nuclear Option: If All Else Fails

If you're still stuck after Week 1:

1. **Start from official PyTorch image**
   ```dockerfile
   FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime
   ```
   
2. **Don't build mmcv from source**
   - Use `pip install mmcv==2.1.0` (no CUDA ops)
   - Accept slower performance
   - Skip mmpose entirely
   
3. **Get basic system working first**
   - MuseTalk + PersonaPlex only
   - No pose estimation
   - Validate this works
   
4. **Add mmpose as separate layer LATER**

Sometimes "good enough" ships while "perfect" never does.
