# Quick Migration Guide

## What You Have Now vs What You Need

### Current Files (BROKEN)
```
onebox-deploy/
├── Dockerfile.multistage          ❌ Multiple pip stages, fake build args
├── build_onebox.sh                 ❌ Build args not used by Dockerfile
├── start_services.sh               ❌ Hardcoded runtime variables
└── dependency_logs.txt             ❌ Shows ABI mismatch errors
```

### New Files (FIXED)
```
EXPERT_DEPENDENCY_STRATEGY.md       📘 Complete strategy document
Dockerfile.fixed                    ✅ Single venv, smoke tests, actual ARG usage
build_onebox.improved.sh            ✅ Validation, freezing, commit SHA support
start_services.fixed.sh             ✅ Respects ALL environment variables
inspect-dependencies.sh             🔍 Debugging and auditing tool
```

---

## Quick Start: Fix Your Build TODAY

### Step 1: Replace Your Files (5 minutes)

```bash
# Backup your current setup
cp Dockerfile.multistage Dockerfile.multistage.backup
cp build_onebox.sh build_onebox.sh.backup
cp start_services.sh start_services.sh.backup

# Use the fixed versions
cp Dockerfile.fixed Dockerfile.multistage
cp build_onebox.improved.sh build_onebox.sh
cp start_services.fixed.sh start_services.sh
chmod +x build_onebox.sh start_services.sh inspect-dependencies.sh
```

### Step 2: Build with Validation (30 minutes)

```bash
# Build with smoke tests enabled
./build_onebox.sh
```

**Expected output:**
```
=== Running Build-Time Smoke Tests ===
Testing PyTorch stack...
  torch: 2.9.0+cu130
  CUDA: 13.0
  ✓ torchaudio._extension loads
Testing MMCV stack...
  mmcv: 2.1.0
  ✓ mmcv._ext loads (CUDA ops available)
  mmdet: 3.2.0
  mmpose: 1.3.2

=== ALL CRITICAL IMPORTS SUCCESSFUL ===
```

**If build FAILS:**
- Smoke tests caught the problem BEFORE you got a broken image
- Check the error message (it will tell you which import failed)
- This is GOOD - you want failures at build time, not runtime

### Step 3: Test the Image (2 minutes)

```bash
# Quick smoke test
docker run --rm musetalk-onebox:blackwell python -c "import mmcv._ext; print('✓ OK')"

# Full audit
./inspect-dependencies.sh smoke musetalk-onebox:blackwell
```

### Step 4: Pin to Commit SHAs (5 minutes)

```bash
# Get current commits
MUSETALK_SHA=$(docker run --rm musetalk-onebox:blackwell sh -c 'cd /opt/musetalk && git rev-parse HEAD')
PERSONAPLEX_SHA=$(docker run --rm musetalk-onebox:blackwell sh -c 'cd /opt/personaplex && git rev-parse HEAD')

echo "MUSETALK_SHA=${MUSETALK_SHA}"
echo "PERSONAPLEX_SHA=${PERSONAPLEX_SHA}"

# Update build_onebox.sh
# Replace:
#   MUSETALK_REF="${MUSETALK_REF:-main}"
# With:
#   MUSETALK_REF="${MUSETALK_REF:-abc123...}"  # your actual SHA
```

**Why this matters:**
- Yesterday's `main` ≠ today's `main`
- Commit SHAs never change
- Reproducible builds forever

### Step 5: Freeze Dependencies (2 minutes)

```bash
# Extract frozen requirements
./inspect-dependencies.sh freeze musetalk-onebox:blackwell > frozen-$(date +%Y%m%d).txt

# Check critical versions
grep -E '^(torch|mmcv|mmdet|mmpose)==' frozen-*.txt

# Commit to git
git add frozen-*.txt
git commit -m "Pin dependencies for reproducible builds"
```

---

## Debugging Checklist

### Problem: Build fails during smoke tests

**What this means:** Critical imports are broken (good we caught it!)

**Check:**
```bash
# See which import failed
docker logs <container_id> | grep -A5 "Running Build-Time Smoke Tests"

# If it's mmcv._ext:
#   → ABI mismatch (torch version changed after mmcv was compiled)
# If it's torchaudio._extension:
#   → Same issue, different library
```

**Fix:**
1. Make sure Dockerfile.fixed is being used (not old Dockerfile.multistage)
2. Clean build from scratch: `docker builder prune -af`
3. Check build args are correct: `grep TORCH_VERSION build_onebox.sh`

### Problem: Container crashes at runtime with "undefined symbol"

**What this means:** You're using the OLD Dockerfile that has multiple pip stages

**Fix:**
```bash
# Verify you're using the fixed Dockerfile
grep "COPY COMPLETE virtual environment" Dockerfile.multistage
# Should see this line. If not, you're using the old file.

# Rebuild from scratch
docker builder prune -af
./build_onebox.sh
```

### Problem: Environment variables not working (MUSE_FPS, etc.)

**What this means:** You're using the OLD start_services.sh with hardcoded values

**Fix:**
```bash
# Verify you're using the fixed version
grep "export FPS=\"\${MUSE_FPS}\"" start_services.sh
# Should see this line. If not, you're using the old file.

# Update the container
cp start_services.fixed.sh start_services.sh
docker build -t musetalk-onebox:blackwell .
```

---

## Common Questions

### Q: Why do I need to rebuild from scratch?

**A:** Docker layer cache can hide problems:
- Old layer has broken mmcv compiled against Torch 2.8
- New layer installs Torch 2.9
- Cache uses old mmcv layer + new torch layer = ABI mismatch

**Solution:** `docker builder prune -af` before builds

### Q: How do I know if my build is reproducible?

**A:** Run this test:
```bash
# Build once
./build_onebox.sh
./inspect-dependencies.sh freeze musetalk-onebox:blackwell > freeze1.txt

# Build again (different day)
docker builder prune -af
./build_onebox.sh
./inspect-dependencies.sh freeze musetalk-onebox:blackwell > freeze2.txt

# Compare
diff freeze1.txt freeze2.txt
# Should be IDENTICAL (or show only timestamps)
```

### Q: Can I skip the mmpose stuff to speed up builds?

**A:** YES! That's what `INSTALL_MMPOSE=0` is for:
```bash
INSTALL_MMPOSE=0 ./build_onebox.sh
```

**Impact:**
- Build time: 30 min → 10 min
- Image size: ~8GB → ~5GB
- Quality: Slight drop (uses face detector fallback)

**For most use cases, the fallback is fine.**

### Q: How do I compare two images to see what changed?

**A:** Use the inspection tool:
```bash
./inspect-dependencies.sh diff \
  musetalk-onebox:2024-01-15 \
  musetalk-onebox:latest
```

Shows exactly which packages changed versions.

---

## File Reference

### EXPERT_DEPENDENCY_STRATEGY.md
- **What:** Complete analysis and strategy document
- **Read:** Before implementing anything
- **Use:** Reference when you hit issues

### Dockerfile.fixed
- **What:** Fixed multi-stage Dockerfile
- **Replaces:** `Dockerfile.multistage`
- **Key fixes:**
  - Single venv (no multiple pip stages)
  - Actual ARG usage (not hardcoded versions)
  - Build-time smoke tests
  - Protected package list

### build_onebox.improved.sh
- **What:** Fixed build script
- **Replaces:** `build_onebox.sh`
- **Key fixes:**
  - Build args actually passed to Dockerfile
  - Automatic dependency freezing
  - Commit SHA pinning support
  - Validation and smoke tests

### start_services.fixed.sh
- **What:** Fixed startup script
- **Replaces:** `start_services.sh`
- **Key fixes:**
  - Uses MUSE_* variables (not hardcoded)
  - Cleaner environment handling
  - Better error messages

### inspect-dependencies.sh
- **What:** Debugging and auditing tool
- **Usage:**
  ```bash
  ./inspect-dependencies.sh freeze <image>      # Extract deps
  ./inspect-dependencies.sh diff <img1> <img2>  # Compare
  ./inspect-dependencies.sh smoke <image>       # Test
  ./inspect-dependencies.sh audit <image>       # Full report
  ```

---

## The 5-Minute Emergency Fix

If you just need it working RIGHT NOW:

```bash
# 1. Add smoke tests to your EXISTING Dockerfile
echo '
RUN python -c "import mmcv._ext; print(\"✓ OK\")" || exit 1
RUN python -c "import torchaudio._extension; print(\"✓ OK\")" || exit 1
' >> Dockerfile.multistage

# 2. Build
docker build -t test .

# If build FAILS → you have ABI mismatch (expected)
# If build PASSES → your image is good (ship it)
```

This won't fix the root cause, but it will STOP you from shipping broken images.

Then schedule time to do the proper migration.

---

## What Success Looks Like

### Before (Current State)
- ❌ Build succeeds but container crashes
- ❌ "It worked yesterday" but not today
- ❌ Can't reproduce builds
- ❌ Environment variables ignored
- ❌ Download dependencies at runtime

### After (Expert State)
- ✅ Build fails if imports are broken (catches problems early)
- ✅ Same build tomorrow = same result
- ✅ Can reproduce any build from git history
- ✅ Environment variables work as expected
- ✅ Hermetic container (no network at runtime)

---

## Support / Troubleshooting

### Still stuck?

1. **Run full audit:**
   ```bash
   ./inspect-dependencies.sh audit musetalk-onebox:blackwell > report.txt
   ```

2. **Check the strategy doc:**
   ```bash
   grep -A10 "Problem: <your issue>" EXPERT_DEPENDENCY_STRATEGY.md
   ```

3. **Verify you're using fixed files:**
   ```bash
   grep "COPY COMPLETE virtual environment" Dockerfile.multistage
   grep "export FPS=" start_services.sh
   ```

### Quick diagnostic:

```bash
# What's actually in your container?
docker run --rm musetalk-onebox:blackwell pip freeze | grep -E '^(torch|mmcv)'

# What versions were built together?
docker run --rm musetalk-onebox:blackwell python -c "
import torch, mmcv
print(f'torch: {torch.__version__}')
print(f'mmcv: {mmcv.__version__}')
try:
    import mmcv._ext
    print('mmcv._ext: OK')
except Exception as e:
    print(f'mmcv._ext: FAIL ({e})')
"
```

If mmcv._ext fails → ABI mismatch → you're still using the old multi-stage pip approach.

---

## Remember

**The goal is not to fix each error as it appears.**

**The goal is to build a system where classes of errors can't happen.**

That's why we:
- Build Python env ONCE (can't have version conflicts)
- Add smoke tests (can't ship broken builds)
- Pin commits (can't have "worked yesterday" bugs)
- Freeze deps (can't have surprise updates)
- Validate builds (can't deploy untested images)

This is how experts build containers. Not "try and see", but "design to prevent".
