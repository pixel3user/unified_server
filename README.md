# Expert Container Dependency Management Solution

## The Problem You Had

Your Docker build succeeded, but containers crashed with:
```
OSError: libtorchaudio.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
```

**Root cause:** Multiple `pip install` stages in the final image allowed PyTorch to be reinstalled AFTER mmcv was compiled, creating an ABI mismatch.

## The Solution

Complete rebuild of your container architecture following expert practices:

1. **Single-stage Python environment** - Build venv once, copy to runtime
2. **Build-time validation** - Smoke tests catch broken imports during build
3. **Dependency locking** - Freeze and pin all versions
4. **Hermetic containers** - No runtime network dependencies
5. **Reproducible builds** - Commit SHAs, not branch names

## Files in This Package

### 📘 Documentation

| File | Description | Read This When |
|------|-------------|----------------|
| `EXPERT_DEPENDENCY_STRATEGY.md` | Complete analysis, strategy, and implementation roadmap | Before you start |
| `MIGRATION_GUIDE.md` | Quick start guide and troubleshooting | You want to fix it NOW |
| `README.md` | This file | You want an overview |

### ✅ Fixed Implementation Files

| File | Replaces | Key Improvements |
|------|----------|------------------|
| `Dockerfile.fixed` | `Dockerfile.multistage` | Single venv, smoke tests, actual ARG usage |
| `build_onebox.improved.sh` | `build_onebox.sh` | Validation, freezing, SHA pinning |
| `start_services.fixed.sh` | `start_services.sh` | Respects environment variables |

### 🔍 Tools

| File | Purpose | Usage |
|------|---------|-------|
| `inspect-dependencies.sh` | Debug and audit builds | `./inspect-dependencies.sh smoke <image>` |

## Quick Start (15 minutes)

### 1. Replace Your Files
```bash
cd your-onebox-deploy-directory
cp Dockerfile.multistage Dockerfile.multistage.backup
cp Dockerfile.fixed Dockerfile.multistage
cp build_onebox.improved.sh build_onebox.sh
cp start_services.fixed.sh start_services.sh
chmod +x build_onebox.sh start_services.sh inspect-dependencies.sh
```

### 2. Build with Validation
```bash
./build_onebox.sh
```

**Expected output:**
```
=== Running Build-Time Smoke Tests ===
Testing PyTorch stack...
  ✓ torch 2.9.0+cu130
Testing MMCV stack...
  ✓ mmcv._ext loads (CUDA ops available)
=== ALL CRITICAL IMPORTS SUCCESSFUL ===
```

### 3. Test the Image
```bash
docker run --rm musetalk-onebox:blackwell python -c "import mmcv._ext; print('✓ OK')"
```

### 4. Pin to Commit SHAs (Production)
```bash
# Get current commits
MUSETALK_SHA=$(docker run --rm musetalk-onebox:blackwell sh -c 'cd /opt/musetalk && git rev-parse HEAD')
PERSONAPLEX_SHA=$(docker run --rm musetalk-onebox:blackwell sh -c 'cd /opt/personaplex && git rev-parse HEAD')

# Update build_onebox.sh with these SHAs
# Replace MUSETALK_REF="${MUSETALK_REF:-main}" with the actual SHA
```

## What Changed?

### Before (Broken)
```dockerfile
# Stage 1: Builder
RUN pip install torch==2.9.0
RUN pip wheel mmcv  # compiled against torch 2.9.0

# Stage 2: Runtime  
RUN pip install torch==2.9.0  # might get 2.9.0 or 2.9.1
RUN pip install /wheels/mmcv*.whl  # compiled against 2.9.0
RUN pip install -r requirements.txt  # might update torch again!
# → ABI mismatch
```

### After (Fixed)
```dockerfile
# Stage 1: Builder
RUN python -m venv /opt/venv
RUN pip install torch==2.9.0
RUN pip wheel mmcv && pip install mmcv*.whl
RUN pip install -r requirements.txt
# All in same venv, no conflicts possible

# Stage 2: Runtime
COPY --from=builder /opt/venv /opt/venv
# Zero pip install operations
# → Can't have ABI mismatch
```

## Key Improvements

### 1. Build-Time Validation
**Old:** Build succeeds, container crashes  
**New:** Build fails if imports are broken

```dockerfile
RUN python -c "import mmcv._ext" || exit 1
```

### 2. Environment Variables
**Old:** Hardcoded values in start_services.sh  
**New:** Respects all MUSE_* variables

```bash
# Old (ignored your MUSE_FPS setting)
FPS=25 BATCH_SIZE=8 ./run_script.sh

# New (uses your MUSE_FPS)
FPS="${MUSE_FPS}" BATCH_SIZE="${MUSE_BATCH_SIZE}" ./run_script.sh
```

### 3. Reproducible Builds
**Old:** `MUSETALK_REF=main` → different code every day  
**New:** `MUSETALK_REF=abc123...` → same code forever

### 4. Dependency Inspection
**Old:** No way to see what changed between builds  
**New:** `./inspect-dependencies.sh diff image1 image2`

### 5. Protected Packages
**Old:** Installing app requirements could overwrite torch  
**New:** Protected package list prevents replacements

```bash
# Protected from being replaced
torch
torchvision
torchaudio
mmcv
mmdet
mmpose
```

## Debugging Guide

### Build fails during smoke tests
**Good!** This means we caught a problem before shipping a broken image.

Check which import failed:
```bash
docker logs <container> | grep -A5 "Smoke Tests"
```

### Container still crashes with undefined symbol
You're using the old Dockerfile. Verify:
```bash
grep "COPY COMPLETE virtual environment" Dockerfile.multistage
# Should find this line. If not, replace with Dockerfile.fixed
```

### Environment variables don't work
You're using the old start_services.sh. Verify:
```bash
grep "export FPS=\"\${MUSE_FPS}\"" start_services.sh
# Should find this line. If not, replace with start_services.fixed.sh
```

## Tools Reference

### Dependency Inspection
```bash
# Extract frozen requirements
./inspect-dependencies.sh freeze musetalk-onebox:latest > frozen.txt

# Compare two images
./inspect-dependencies.sh diff image1 image2

# Run smoke tests
./inspect-dependencies.sh smoke musetalk-onebox:latest

# Full audit report
./inspect-dependencies.sh audit musetalk-onebox:latest > report.txt
```

## What You Get

✅ **Deterministic builds** - Same input = same output  
✅ **Early failure detection** - Broken imports fail at build time  
✅ **Hermetic containers** - No runtime network dependencies  
✅ **Debuggable** - Tools to inspect and compare builds  
✅ **Reproducible** - Can rebuild any version from git history  

## Implementation Phases

### Week 1: Emergency Fix
- [ ] Replace Dockerfile with Dockerfile.fixed
- [ ] Add smoke tests
- [ ] Get one clean build

### Week 2: Production Hardening
- [ ] Pin commit SHAs
- [ ] Freeze dependencies
- [ ] Remove runtime downloads
- [ ] Fix runtime variables

### Week 3: Advanced
- [ ] Split core/pose images
- [ ] Add CI/CD integration
- [ ] Monitoring and metrics

## Support

1. **Read MIGRATION_GUIDE.md** - Quick start and troubleshooting
2. **Read EXPERT_DEPENDENCY_STRATEGY.md** - Complete strategy and rationale
3. **Use inspect-dependencies.sh** - Debug build issues

## The Expert Mindset

You said: "I am tired of solving each error."

The expert approach is not to fix errors one at a time.

**It's to build a system where classes of errors can't happen.**

That's what these files give you:
- ABI mismatches **can't happen** (single venv)
- Broken images **can't ship** (smoke tests)
- "Worked yesterday" bugs **can't happen** (pinned commits)
- Version conflicts **can't happen** (frozen constraints)

This is how you stop fighting dependency hell forever.

---

**Good luck! 🚀**
