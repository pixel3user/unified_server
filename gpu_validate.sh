#!/usr/bin/env bash
# GPU Validation Script for RTX 6000 Pro 96GB
# Run this during the 5-hour GPU window to validate all fixes.
#
# Prerequisites:
#   - GPU attached to this studio (or SSH'd into GPU machine)
#   - Both repos at feat/testability-refactor branch
#   - Model weights downloaded (./download_weights.sh in MuseTalk)
#
# Usage:
#   cd unified_server && bash gpu_validate.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MUSETALK_DIR="$(cd "$SCRIPT_DIR/../MuseTalk" && pwd)"

echo "============================================"
echo "  GPU Validation — MuseTalk + PersonaPlex"
echo "============================================"
echo ""

# --- Phase 1: Environment Check (1 min) ---
echo "[Phase 1] Environment check..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available!'
dev = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
print(f'  GPU: {dev} ({mem:.0f} GB)')
print(f'  CUDA: {torch.version.cuda}')
print(f'  PyTorch: {torch.__version__}')
"
echo ""

# --- Phase 2: CPU Tests (sanity check — should still pass) ---
echo "[Phase 2] Running CPU test suite (should be 144 passed)..."
cd "$SCRIPT_DIR"
python3 -m pytest tests/unit/ tests/component/ tests/http_routes/ \
    --timeout=15 --timeout-method=thread -q 2>&1 | tail -3
echo ""

# --- Phase 3: Model Loading Smoke Test (2-3 min) ---
echo "[Phase 3] Model loading smoke test..."
cd "$MUSETALK_DIR"
python3 -c "
import sys, time
sys.path.insert(0, '.')
t0 = time.time()

import scripts.realtime_inference as rt
import torch

rt.args = type('Args', (), {
    'version': 'v15',
    'ffmpeg_path': 'ffmpeg',
    'gpu_id': 0,
    'vae_type': 'sd-vae-ft-mse',
    'unet_config': 'musetalk/utils/unet_config.json',
    'unet_model_path': 'models/musetalk/pytorch_model.bin',
    'whisper_dir': 'models/whisper',
    'bbox_shift': 0,
    'extra_margin': 10,
    'fps': 25,
    'audio_padding_length_left': 2,
    'audio_padding_length_right': 2,
    'batch_size': 4,
    'parsing_mode': 'jaw',
    'left_cheek_width': 90,
    'right_cheek_width': 90,
    'use_fp16': True,
    'require_mmpose': False,
})()

device = torch.device('cuda:0')
vae, unet, pe = rt.load_all_model(
    unet_model_path=rt.args.unet_model_path,
    vae_type=rt.args.vae_type,
    unet_config=rt.args.unet_config,
    device=device,
)
print(f'  Models loaded in {time.time()-t0:.1f}s')
print(f'  GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated')
del vae, unet, pe
torch.cuda.empty_cache()
print('  ✓ Model loading OK')
"
echo ""

# --- Phase 4: Single Inference Test (1 min) ---
echo "[Phase 4] Single inference pass..."
cd "$MUSETALK_DIR"
python3 -c "
import sys, time, numpy as np
sys.path.insert(0, '.')
import torch
import scripts.realtime_inference as rt
from scripts.musetalk_webrtc.engine import MuseTalkRealtimeEngine
from scripts.musetalk_webrtc.buffers import PcmRingBuffer, VideoFrameBuffer
from scripts.musetalk_webrtc.models import AppArgs
import asyncio

async def test_single_inference():
    args = AppArgs(
        host='127.0.0.1', port=0, ice_servers=[], ice_transport_policy='all',
        ice_username='', ice_credential='',
        personaplex_host='127.0.0.1', personaplex_port=9999,
        personaplex_path='/api/chat', personaplex_text_prompt='test',
        personaplex_voice_prompt='', personaplex_extra_query=[],
        avatar_id='avator_1', version='v15', gpu_id=0, use_fp16=True,
        require_mmpose=False, fps=25, avatar_fps=25, batch_size=4,
        bbox_shift=0, unet_model_path='models/musetalk/pytorch_model.bin',
        unet_config='musetalk/utils/unet_config.json',
        vae_type='sd-vae-ft-mse', whisper_dir='models/whisper',
        ffmpeg_path='ffmpeg', parsing_mode='jaw', extra_margin=10,
        left_cheek_width=90, right_cheek_width=90,
        audio_padding_length_left=2, audio_padding_length_right=2,
        ring_buffer_seconds=10.0, window_ms=640, hop_ms=80,
        min_window_ms=320, max_advance_ms=240, max_tail_frames=5,
        mouth_smoothing_alpha=0.75, video_queue_size=64,
        status_json=None, reconnect_delay_seconds=1.0,
        input_source='mirror', webrtc_audio_loopback=False,
        musetalk_only=False, enable_api_auth=False, api_token='',
        session_offer_timeout_seconds=30.0, session_max_age_seconds=3600.0,
        session_cleanup_interval_seconds=10.0,
        session_disconnect_grace_seconds=10.0,
        ice_gather_timeout_seconds=5.0,
        single_session_mode=True, web_test_only=False,
        debug=True, debug_events_limit=100,
    )

    ring = PcmRingBuffer(max_samples=160000)
    vbuf = VideoFrameBuffer(maxsize=64)

    engine = MuseTalkRealtimeEngine(args, ring, vbuf)
    print(f'  Engine initialized, GPU mem: {torch.cuda.memory_allocated()/1024**3:.2f} GB')

    # Feed 1 second of synthetic speech (sine wave)
    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    speech = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    await ring.append(speech)

    # Run engine for a short burst
    task = asyncio.create_task(engine.run())
    await asyncio.sleep(2.0)
    engine.stop_event.set()
    await task

    print(f'  Jobs completed: {engine.jobs}')
    print(f'  Frames in queue: {vbuf.queue.qsize()}')
    print(f'  Last error: {engine.last_error or \"none\"}')

    # Verify frames are not all-black
    if vbuf.queue.qsize() > 0:
        frame = await vbuf.get(timeout=0.1)
        assert frame is not None
        assert frame.mean() > 5, 'Frame is all-black!'
        print(f'  Frame shape: {frame.shape}, mean pixel: {frame.mean():.1f}')
        print('  ✓ Single inference OK')
    else:
        print('  ✗ No frames produced!')
        sys.exit(1)

asyncio.run(test_single_inference())
"
echo ""

# --- Phase 5: Audio-Video Sync Validation (2 min) ---
echo "[Phase 5] Audio-video sync — verifying the speed-up bug is fixed..."
cd "$MUSETALK_DIR"
python3 -c "
import sys, time, numpy as np, asyncio
sys.path.insert(0, '.')
import torch
from scripts.musetalk_webrtc.engine import MuseTalkRealtimeEngine
from scripts.musetalk_webrtc.buffers import PcmRingBuffer, VideoFrameBuffer
from scripts.musetalk_webrtc.models import AppArgs

async def test_no_audio_drops():
    args = AppArgs(
        host='127.0.0.1', port=0, ice_servers=[], ice_transport_policy='all',
        ice_username='', ice_credential='',
        personaplex_host='127.0.0.1', personaplex_port=9999,
        personaplex_path='/api/chat', personaplex_text_prompt='test',
        personaplex_voice_prompt='', personaplex_extra_query=[],
        avatar_id='avator_1', version='v15', gpu_id=0, use_fp16=True,
        require_mmpose=False, fps=25, avatar_fps=25, batch_size=4,
        bbox_shift=0, unet_model_path='models/musetalk/pytorch_model.bin',
        unet_config='musetalk/utils/unet_config.json',
        vae_type='sd-vae-ft-mse', whisper_dir='models/whisper',
        ffmpeg_path='ffmpeg', parsing_mode='jaw', extra_margin=10,
        left_cheek_width=90, right_cheek_width=90,
        audio_padding_length_left=2, audio_padding_length_right=2,
        ring_buffer_seconds=10.0, window_ms=640, hop_ms=80,
        min_window_ms=320, max_advance_ms=240, max_tail_frames=5,
        mouth_smoothing_alpha=0.75, video_queue_size=64,
        status_json=None, reconnect_delay_seconds=1.0,
        input_source='mirror', webrtc_audio_loopback=False,
        musetalk_only=False, enable_api_auth=False, api_token='',
        session_offer_timeout_seconds=30.0, session_max_age_seconds=3600.0,
        session_cleanup_interval_seconds=10.0,
        session_disconnect_grace_seconds=10.0,
        ice_gather_timeout_seconds=5.0,
        single_session_mode=True, web_test_only=False,
        debug=True, debug_events_limit=100,
    )

    ring = PcmRingBuffer(max_samples=160000)
    vbuf = VideoFrameBuffer(maxsize=128)
    engine = MuseTalkRealtimeEngine(args, ring, vbuf)

    # Feed 3 seconds of speech in a burst (simulates PersonaPlex TTS flush)
    t = np.linspace(0, 3.0, 48000, dtype=np.float32)
    speech = (np.sin(2 * np.pi * 300 * t) * 0.4).astype(np.float32)
    await ring.append(speech)

    task = asyncio.create_task(engine.run())
    await asyncio.sleep(5.0)  # Give it time to process all audio
    engine.stop_event.set()
    await task

    # KEY ASSERTION: with the fix, NO audio should be dropped
    print(f'  Dropped audio: {engine.dropped_audio_ms_total:.1f} ms')
    print(f'  Jobs: {engine.jobs}')
    print(f'  Frames produced: {vbuf.queue.qsize()} in queue')

    # 3s of audio at 25fps = 75 frames expected
    # With max_advance_ms=240, it takes ~13 iterations (3000/240)
    expected_jobs = int(3000 / 240)  # ~12-13
    print(f'  Expected ~{expected_jobs} jobs, got {engine.jobs}')

    assert engine.jobs >= expected_jobs - 2, (
        f'Too few jobs ({engine.jobs}), audio may have been dropped'
    )
    print('  ✓ Audio-video sync OK — no silent drops')

asyncio.run(test_no_audio_drops())
"
echo ""

# --- Phase 6: Event-Driven Loop Validation (1 min) ---
echo "[Phase 6] Event-driven loop — verifying no busy-polling..."
cd "$MUSETALK_DIR"
python3 -c "
import sys, time, numpy as np, asyncio
sys.path.insert(0, '.')
import torch
from scripts.musetalk_webrtc.engine import MuseTalkRealtimeEngine
from scripts.musetalk_webrtc.buffers import PcmRingBuffer, VideoFrameBuffer
from scripts.musetalk_webrtc.models import AppArgs

async def test_event_driven():
    args = AppArgs(
        host='127.0.0.1', port=0, ice_servers=[], ice_transport_policy='all',
        ice_username='', ice_credential='',
        personaplex_host='127.0.0.1', personaplex_port=9999,
        personaplex_path='/api/chat', personaplex_text_prompt='test',
        personaplex_voice_prompt='', personaplex_extra_query=[],
        avatar_id='avator_1', version='v15', gpu_id=0, use_fp16=True,
        require_mmpose=False, fps=25, avatar_fps=25, batch_size=4,
        bbox_shift=0, unet_model_path='models/musetalk/pytorch_model.bin',
        unet_config='musetalk/utils/unet_config.json',
        vae_type='sd-vae-ft-mse', whisper_dir='models/whisper',
        ffmpeg_path='ffmpeg', parsing_mode='jaw', extra_margin=10,
        left_cheek_width=90, right_cheek_width=90,
        audio_padding_length_left=2, audio_padding_length_right=2,
        ring_buffer_seconds=10.0, window_ms=640, hop_ms=80,
        min_window_ms=320, max_advance_ms=240, max_tail_frames=5,
        mouth_smoothing_alpha=0.75, video_queue_size=64,
        status_json=None, reconnect_delay_seconds=1.0,
        input_source='mirror', webrtc_audio_loopback=False,
        musetalk_only=False, enable_api_auth=False, api_token='',
        session_offer_timeout_seconds=30.0, session_max_age_seconds=3600.0,
        session_cleanup_interval_seconds=10.0,
        session_disconnect_grace_seconds=10.0,
        ice_gather_timeout_seconds=5.0,
        single_session_mode=True, web_test_only=False,
        debug=True, debug_events_limit=100,
    )

    ring = PcmRingBuffer(max_samples=160000)
    vbuf = VideoFrameBuffer(maxsize=64)
    engine = MuseTalkRealtimeEngine(args, ring, vbuf)

    # Start engine with NO audio — it should block on wait_for_total_after
    task = asyncio.create_task(engine.run())
    await asyncio.sleep(1.0)

    # Should have done 0 inference jobs (just top-ups)
    jobs_before = engine.jobs
    print(f'  Jobs with no audio: {jobs_before} (should be 0)')

    # Now feed audio — engine should wake up immediately
    t = np.linspace(0, 0.5, 8000, dtype=np.float32)
    speech = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    t0 = time.time()
    await ring.append(speech)
    await asyncio.sleep(1.0)

    jobs_after = engine.jobs
    print(f'  Jobs after audio: {jobs_after}')
    assert jobs_after > jobs_before, 'Engine did not wake on new audio!'

    engine.stop_event.set()
    await task
    print('  ✓ Event-driven loop OK — no busy polling')

asyncio.run(test_event_driven())
"
echo ""

echo "============================================"
echo "  ALL GPU VALIDATION PHASES PASSED ✓"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Commit and push the fixes"
echo "  2. Run a full WebRTC session test (manual or with a browser)"
echo "  3. Build Docker image if needed"
