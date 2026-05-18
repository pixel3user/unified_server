#!/usr/bin/env python
import inspect
import json
import os
import hashlib
import ssl
import time
import types
import urllib.error
import urllib.request
import asyncio

from aiohttp import web
from huggingface_hub import hf_hub_download
import sentencepiece
import torch

from moshi.models import loaders as personaplex_loaders
from moshi.server import ServerState as PersonaPlexState
from moshi.server import _get_voice_prompt_dir, seed_all, torch_auto_device
from scripts.musetalk_webrtc.cli import parse_args as parse_musetalk_args
import scripts.musetalk_webrtc.server as musetalk_server_mod

WebRtcApp = musetalk_server_mod.WebRtcApp


CUSTOM_HTML_PAGE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MuseTalk WebRTC</title>
</head>
<body style="margin:0;background:#121212;color:#e5e5e5;font-family:system-ui,sans-serif">
  <div style="padding:12px;font-size:14px">MuseTalk In-Memory WebRTC Preview</div>
  <div style="display:flex;gap:12px;padding:0 12px 12px 12px;flex-wrap:wrap">
    <video id="v" autoplay playsinline muted style="width:min(96vw,960px);background:black;border-radius:10px"></video>
    <audio id="a" autoplay controls style="width:min(96vw,960px)"></audio>
  </div>
  <div style="padding:0 12px 12px 12px">
    <button id="start">Start</button>
    <button id="stop" disabled>Stop</button>
    <span id="state" style="margin-left:8px">idle</span>
    <div id="pp-status" style="margin-top:8px;font-size:12px;color:#ffd37a">PersonaPlex: idle</div>
    <div id="mic" style="margin-top:8px;font-size:12px;color:#8fd48f">mic: idle</div>
    <div style="margin-top:4px;height:10px;width:100%;max-width:300px;background:#333;border-radius:5px;overflow:hidden;">
      <div id="mic-level" style="height:100%;width:0%;background:#4caf50;transition:width 0.1s;"></div>
    </div>
    <div id="dbg" style="margin-top:8px;font-size:12px;white-space:pre-wrap;color:#b0b0b0"></div>
    <div id="log" style="margin-top:8px;font-size:12px;white-space:pre-wrap;color:#d0d0d0;background:#1b1b1b;border-radius:8px;padding:10px;max-width:960px;max-height:220px;overflow:auto"></div>
  </div>
<script>
let pc = null;
let localStream = null;
let sessionId = null;
let sessionToken = null;
let audioContext = null;
let analyser = null;
let latestStatus = null;
let latestRuntime = null;

const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');
const stateEl = document.getElementById('state');
const micEl = document.getElementById('mic');
const micLevelEl = document.getElementById('mic-level');
const ppStatusEl = document.getElementById('pp-status');
const dbgEl = document.getElementById('dbg');
const logEl = document.getElementById('log');

function setState(next) {
  stateEl.textContent = next;
}

function setMic(next) {
  micEl.textContent = 'mic: ' + next;
}

function setPersonaPlexStatus(next) {
  ppStatusEl.textContent = 'PersonaPlex: ' + next;
}

function formatTs(tsEpoch) {
  if (!tsEpoch) return '--:--:--';
  const d = new Date(tsEpoch * 1000);
  return d.toLocaleTimeString();
}

function renderLogs(statusPayload, runtimePayload) {
  const debugEvents = (((statusPayload || {}).debug || {}).events || []).slice(-12);
  const runtimeEvents = ((runtimePayload || {}).events || []).slice(-12);
  const merged = debugEvents.concat(runtimeEvents).sort((a, b) => (a.ts_epoch || 0) - (b.ts_epoch || 0)).slice(-18);
  if (!merged.length) {
    logEl.textContent = 'No runtime events yet.';
    return;
  }
  logEl.textContent = merged.map((ev) => {
    const parts = [];
    parts.push('[' + formatTs(ev.ts_epoch) + ']');
    parts.push(ev.event || 'event');
    const fields = Object.assign({}, ev);
    delete fields.ts_epoch;
    delete fields.event;
    const extra = Object.entries(fields).map(([k, v]) => k + '=' + JSON.stringify(v)).join(' ');
    if (extra) parts.push(extra);
    return parts.join(' ');
  }).join('\\n');
}

function computePersonaPlexStatus(statusPayload, runtimePayload) {
  const status = statusPayload || {};
  const runtime = runtimePayload || {};
  const debugEvents = (((status.debug || {}).events) || []);
  const lastEvent = debugEvents.length ? debugEvents[debugEvents.length - 1] : null;

  if (!(status.personaplex || {}).duplex_chat_mode) {
    return 'mirror mode';
  }
  if (runtime.last_error) {
    return 'error: ' + runtime.last_error;
  }
  if (runtime.voice_loading) {
    return 'loading voice ' + (runtime.voice_prompt || '');
  }
  if ((runtime.active_chat_requests || 0) > 0 && !runtime.handshake_ready) {
    return 'preparing voice and prompts';
  }
  if (lastEvent && lastEvent.event === 'personaplex_chat.starting') {
    return 'starting chat bridge';
  }
  if (lastEvent && lastEvent.event === 'personaplex_chat.started') {
    return 'chat bridge started';
  }
  if (lastEvent && lastEvent.event === 'personaplex_chat.ws_connected') {
    return 'connected to PersonaPlex';
  }
  if (lastEvent && lastEvent.event === 'personaplex_chat.handshake') {
    return 'voice ready';
  }
  if (lastEvent && (lastEvent.event === 'personaplex_chat.rx_audio' || lastEvent.event === 'personaplex_chat.tx_audio')) {
    return 'speaking';
  }
  if ((status.webrtc || {}).session_count > 0) {
    return 'waiting for voice';
  }
  return 'idle';
}

async function waitIceGatheringComplete(pc, timeoutMs = 4000) {
  if (pc.iceGatheringState === 'complete') return;
  await new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      pc.removeEventListener('icegatheringstatechange', onState);
      resolve();
    };
    const onState = () => {
      if (pc.iceGatheringState === 'complete') finish();
    };
    pc.addEventListener('icegatheringstatechange', onState);
    setTimeout(finish, timeoutMs);
  });
}

function updateMicLevel() {
  if (!analyser) return;
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(data);
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    const val = (data[i] - 128) / 128;
    sum += val * val;
  }
  let rms = Math.sqrt(sum / data.length);
  let db = 20 * Math.log10(rms || 0.0001);
  let pct = Math.max(0, Math.min(100, (db + 60) * (100/60)));
  micLevelEl.style.width = pct + '%';
  if (localStream) requestAnimationFrame(updateMicLevel);
}

async function cleanupPeer() {
  if (pc) {
    try { pc.ontrack = null; } catch (e) {}
    try { pc.onconnectionstatechange = null; } catch (e) {}
    try { pc.close(); } catch (e) {}
    pc = null;
  }
  if (localStream) {
    for (const track of localStream.getTracks()) {
      try { track.stop(); } catch (e) {}
    }
    localStream = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
    analyser = null;
  }
  micLevelEl.style.width = '0%';
}

async function cleanupSession() {
  if (!sessionId || !sessionToken) {
    sessionId = null;
    sessionToken = null;
    return;
  }
  try {
    await fetch('/v1/sessions/' + sessionId, {
      method: 'DELETE',
      headers: { 'x-session-token': sessionToken },
    });
  } catch (e) {}
  sessionId = null;
  sessionToken = null;
}

async function start() {
  if (pc) {
    await stop();
  }
  startBtn.disabled = true;
  stopBtn.disabled = false;
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Browser does not support getUserMedia');
    }
    setState('requesting-mic');
    setMic('requesting permission');
    setPersonaPlexStatus('waiting for startup');

    const rtcCfg = await (await fetch('/config')).json();
    localStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
      video: false,
    });
    setMic('granted (processed audio)');

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const micSource = audioContext.createMediaStreamSource(localStream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    micSource.connect(analyser);
    requestAnimationFrame(updateMicLevel);

    pc = new RTCPeerConnection(rtcCfg);
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    for (const track of localStream.getAudioTracks()) {
      pc.addTrack(track, localStream);
    }
    pc.onconnectionstatechange = () => {
      if (pc) setState(pc.connectionState);
    };
    pc.ontrack = (ev) => {
      if (ev.track.kind === 'video') {
        document.getElementById('v').srcObject = ev.streams[0];
      } else if (ev.track.kind === 'audio') {
        document.getElementById('a').srcObject = ev.streams[0];
      }
    };

    setState('connecting');
    setPersonaPlexStatus('creating session');
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitIceGatheringComplete(pc, 5000);
    const resp = await fetch('/offer', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
    });
    if (!resp.ok) {
      throw new Error('offer failed: ' + await resp.text());
    }
    const answer = await resp.json();
    sessionId = answer.session_id || null;
    sessionToken = answer.session_token || null;
    await pc.setRemoteDescription(answer);
    setState('connected');
  } catch (e) {
    console.error(e);
    setState('error');
    setMic('error');
    setPersonaPlexStatus('error');
    await cleanupPeer();
    await cleanupSession();
    stopBtn.disabled = true;
  } finally {
    startBtn.disabled = false;
  }
}

async function stop() {
  stopBtn.disabled = true;
  setState('stopping');
  await cleanupPeer();
  await cleanupSession();
  document.getElementById('v').srcObject = null;
  document.getElementById('a').srcObject = null;
  setMic('idle');
  setPersonaPlexStatus('idle');
  setState('idle');
}

async function refreshStatus() {
  try {
    const [statusResp, runtimeResp] = await Promise.all([
      fetch('/status'),
      fetch('/personaplex/runtime'),
    ]);
    latestStatus = await statusResp.json();
    latestRuntime = await runtimeResp.json();
    dbgEl.textContent = JSON.stringify({
      status: latestStatus,
      personaplex_runtime: latestRuntime,
    }, null, 2);
    setPersonaPlexStatus(computePersonaPlexStatus(latestStatus, latestRuntime));
    renderLogs(latestStatus, latestRuntime);
  } catch (e) {
    logEl.textContent = 'status fetch failed: ' + e;
  }
}

startBtn.onclick = start;
stopBtn.onclick = stop;

setInterval(refreshStatus, 1000);
refreshStatus();

window.addEventListener('beforeunload', () => {
  cleanupPeer();
});
</script>
</body>
</html>
"""


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cuda_mem_stats() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    device_index = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    gib = 1024 ** 3
    return {
        "cuda_available": True,
        "device_index": device_index,
        "device_name": torch.cuda.get_device_name(device_index),
        "allocated_gb": round(torch.cuda.memory_allocated(device_index) / gib, 3),
        "reserved_gb": round(torch.cuda.memory_reserved(device_index) / gib, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(device_index) / gib, 3),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved(device_index) / gib, 3),
        "free_gb": round(free_bytes / gib, 3),
        "total_gb": round(total_bytes / gib, 3),
    }


def _log_cuda_mem(stage: str) -> None:
    stats = _cuda_mem_stats()
    if not stats.get("cuda_available"):
        print(f"[gpu] {stage}: cuda unavailable")
        return
    print(
        "[gpu] "
        f"{stage}: device={stats['device_index']} name={stats['device_name']} "
        f"allocated={stats['allocated_gb']}GiB reserved={stats['reserved_gb']}GiB "
        f"max_allocated={stats['max_allocated_gb']}GiB max_reserved={stats['max_reserved_gb']}GiB "
        f"free={stats['free_gb']}GiB total={stats['total_gb']}GiB"
    )


def _new_runtime_state() -> dict:
    return {
        "voice_prompt": None,
        "active_chat_requests": 0,
        "voice_loading": False,
        "handshake_ready": False,
        "last_error": None,
        "events": [],
        "gpu": _cuda_mem_stats(),
    }


def _runtime_event(runtime: dict, event: str, **fields) -> None:
    payload = {"ts_epoch": round(time.time(), 3), "event": event}
    if fields:
        payload.update(fields)
    runtime["events"].append(payload)
    runtime["events"] = runtime["events"][-50:]


def _patch_frontend() -> None:
    musetalk_server_mod.HTML_PAGE = CUSTOM_HTML_PAGE


class CloudflareTurnProvider:
    def __init__(self, token_id: str, api_token: str, ttl_seconds: int, fallback_policy: str):
        self.token_id = token_id.strip()
        self.api_token = api_token.strip()
        self.ttl_seconds = max(300, int(ttl_seconds))
        self.fallback_policy = fallback_policy
        self._cached_config = None
        self._cached_until = 0.0
        self.last_error = None
        self.last_status_code = None
        self.last_response_excerpt = None

    def debug_identity(self) -> dict:
        token_hash = hashlib.sha256(self.api_token.encode("utf-8")).hexdigest()[:12] if self.api_token else ""
        return {
            "token_id_prefix": self.token_id[:8],
            "token_id_suffix": self.token_id[-6:],
            "api_token_len": len(self.api_token),
            "api_token_sha256_prefix": token_hash,
        }

    def _fetch(self) -> dict:
        payload = json.dumps({"ttl": self.ttl_seconds}).encode("utf-8")
        req = urllib.request.Request(
            url=f"https://rtc.live.cloudflare.com/v1/turn/keys/{self.token_id}/credentials/generate-ice-servers",
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
                # Cloudflare rejects the default Python urllib fingerprint in this
                # environment with HTTP 403 / error code 1010, while curl/aiohttp
                # with an explicit user-agent succeeds.
                "User-Agent": "onebox-turn-probe/1.0",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                self.last_status_code = getattr(resp, "status", None)
                body = resp.read().decode("utf-8")
                self.last_response_excerpt = body[:400]
                data = json.loads(body)
        except urllib.error.HTTPError as exc:
            self.last_status_code = exc.code
            body = exc.read().decode("utf-8", errors="replace")
            self.last_response_excerpt = body[:400]
            raise RuntimeError(
                f"Cloudflare TURN HTTP {exc.code}; response={self.last_response_excerpt!r}; identity={self.debug_identity()}"
            ) from exc

        filtered_servers = []
        for server in data.get("iceServers", []):
            entry = dict(server)
            urls = entry.get("urls", [])
            if isinstance(urls, str):
                urls = [urls]
            urls = [u for u in urls if ":53" not in u]
            entry["urls"] = urls
            filtered_servers.append(entry)

        return {
            "iceServers": filtered_servers,
            "iceTransportPolicy": self.fallback_policy,
        }

    def get_config(self) -> dict:
        now = time.time()
        if self._cached_config is not None and now < self._cached_until:
            return self._cached_config

        config = self._fetch()
        # Refresh before expiry so long calls can renegotiate cleanly.
        refresh_window = min(300, max(60, self.ttl_seconds // 10))
        self._cached_config = config
        self._cached_until = now + self.ttl_seconds - refresh_window
        self.last_error = None
        return config


def _install_cloudflare_turn(musetalk_app_state: WebRtcApp) -> None:
    turn_provider = os.environ.get("TURN_PROVIDER", "").strip().lower()
    if turn_provider and turn_provider not in {"cloudflare", "cloudflare_turn"}:
        print(f"[cloudflare-turn] skipped because TURN_PROVIDER={turn_provider}")
        return

    token_id = os.environ.get("CLOUDFLARE_TURN_TOKEN_ID", "").strip()
    api_token = os.environ.get("CLOUDFLARE_TURN_API_TOKEN", "").strip()
    if not token_id or not api_token:
        return

    browser_only = os.environ.get("CLOUDFLARE_TURN_BROWSER_ONLY", "1").strip().lower() not in {"0", "false", "no"}

    provider = CloudflareTurnProvider(
        token_id=token_id,
        api_token=api_token,
        ttl_seconds=int(os.environ.get("CLOUDFLARE_TURN_TTL_SECONDS", "86400")),
        fallback_policy=os.environ.get("ICE_TRANSPORT_POLICY", musetalk_app_state.args.ice_transport_policy).strip() or "all",
    )
    print(f"[cloudflare-turn] configured identity={provider.debug_identity()} browser_only={browser_only}")
    musetalk_app_state.cloudflare_turn_provider = provider

    original_config = musetalk_app_state.config
    original_config_v1 = musetalk_app_state.config_v1
    original_browser_rtc_config = musetalk_app_state._browser_rtc_config
    original_aiortc_ice_servers = musetalk_app_state._aiortc_ice_servers

    def _cloudflare_browser_rtc_config(self) -> dict:
        try:
            return provider.get_config()
        except Exception as exc:
            provider.last_error = repr(exc)
            print(f"[cloudflare-turn] falling back to static RTC config: {exc!r}")
            return original_browser_rtc_config()

    def _cloudflare_aiortc_ice_servers(self):
        if browser_only:
            # In this deployment the browser benefits from Cloudflare TURN, but
            # the server-side aiortc/aioice leg is more reliable with its
            # original ICE config (typically direct/public candidates or STUN).
            return original_aiortc_ice_servers()
        try:
            rtc_cfg = provider.get_config()
            servers = []
            RTCIceServer = musetalk_server_mod.RTCIceServer
            for entry in rtc_cfg.get("iceServers", []):
                urls = entry.get("urls", [])
                if isinstance(urls, str):
                    urls = [urls]
                for url in urls:
                    kw = {"urls": url}
                    if entry.get("username") and str(url).lower().startswith(("turn:", "turns:")):
                        kw["username"] = entry["username"]
                    if entry.get("credential") and str(url).lower().startswith(("turn:", "turns:")):
                        kw["credential"] = entry["credential"]
                    servers.append(RTCIceServer(**kw))
            return servers
        except Exception as exc:
            provider.last_error = repr(exc)
            print(f"[cloudflare-turn] falling back to static aiortc ICE config: {exc!r}")
            return original_aiortc_ice_servers()

    async def _cloudflare_config(self, request: web.Request):
        try:
            return web.json_response(provider.get_config())
        except Exception as exc:
            provider.last_error = repr(exc)
            print(f"[cloudflare-turn] /config fallback: {exc!r}")
            return await original_config(request)

    async def _cloudflare_config_v1(self, request: web.Request):
        try:
            rtc_cfg = provider.get_config()
            return web.json_response(
                {
                    "rtc_config": rtc_cfg,
                    "single_session_mode": self.args.single_session_mode,
                    "session_token_header": musetalk_server_mod.SESSION_TOKEN_HEADER,
                    "auth_enabled": self.args.enable_api_auth,
                }
            )
        except Exception as exc:
            provider.last_error = repr(exc)
            print(f"[cloudflare-turn] /v1/config fallback: {exc!r}")
            return await original_config_v1(request)

    musetalk_app_state._browser_rtc_config = types.MethodType(_cloudflare_browser_rtc_config, musetalk_app_state)
    musetalk_app_state._aiortc_ice_servers = types.MethodType(_cloudflare_aiortc_ice_servers, musetalk_app_state)
    musetalk_app_state.config = types.MethodType(_cloudflare_config, musetalk_app_state)
    musetalk_app_state.config_v1 = types.MethodType(_cloudflare_config_v1, musetalk_app_state)


def _build_personaplex_state() -> PersonaPlexState:
    hf_repo = os.environ.get("HF_REPO", personaplex_loaders.DEFAULT_REPO)
    device = torch_auto_device(os.environ.get("DEVICE", "cuda"))
    voice_prompt_dir = _get_voice_prompt_dir(os.environ.get("VOICE_PROMPT_DIR"), hf_repo)

    seed_all(42424242)
    _log_cuda_mem("personaplex.start")

    # Match PersonaPlex's normal startup flow so unified mode behaves the same
    # as the standalone server while still sharing a single process.
    hf_hub_download(hf_repo, "config.json")

    print("[unified_server] Loading PersonaPlex models...")
    mimi_weight = os.environ.get("MIMI_WEIGHT") or hf_hub_download(hf_repo, personaplex_loaders.MIMI_NAME)
    mimi = personaplex_loaders.get_mimi(mimi_weight, device)
    _log_cuda_mem("personaplex.after_mimi")
    other_mimi = personaplex_loaders.get_mimi(mimi_weight, device)
    _log_cuda_mem("personaplex.after_other_mimi")

    tokenizer_path = os.environ.get("TOKENIZER") or hf_hub_download(hf_repo, personaplex_loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))

    moshi_weight = os.environ.get("MOSHI_WEIGHT") or hf_hub_download(hf_repo, personaplex_loaders.MOSHI_NAME)
    loader_sig = inspect.signature(personaplex_loaders.get_moshi_lm)
    loader_kwargs = {"device": device}
    optional_loader_flags = {
        "cpu_offload": _env_flag("CPU_OFFLOAD"),
        "lm_int8": _env_flag("LM_INT8"),
        "lm_torchao_int8": _env_flag("LM_TORCHAO_INT8"),
        "lm_torchao_weight_only": _env_flag("LM_TORCHAO_WEIGHT_ONLY"),
    }
    for name, value in optional_loader_flags.items():
        if name in loader_sig.parameters:
            loader_kwargs[name] = value

    lm = personaplex_loaders.get_moshi_lm(moshi_weight, **loader_kwargs)
    lm.eval()
    _log_cuda_mem("personaplex.after_moshi_lm")

    state = PersonaPlexState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=device,
        voice_prompt_dir=voice_prompt_dir,
    )
    print("[unified_server] PersonaPlex models loaded. Warming up...")
    _log_cuda_mem("personaplex.before_warmup")
    state.warmup()
    _log_cuda_mem("personaplex.after_warmup")
    print("[unified_server] PersonaPlex warmup complete.")
    return state


def _install_personaplex_runtime_routes(app: web.Application, personaplex_state: PersonaPlexState) -> None:
    runtime = _new_runtime_state()
    cloudflare_provider = app.get("cloudflare_turn_provider")
    if cloudflare_provider is not None:
        runtime["cloudflare_turn_enabled"] = True
        runtime["cloudflare_turn_last_error"] = None
    app["personaplex_runtime"] = runtime

    original_chat = personaplex_state.handle_chat
    original_avatar_audio = personaplex_state.handle_avatar_audio
    original_voices = getattr(personaplex_state, "handle_voices", None)

    async def chat_with_runtime(request: web.Request):
        voice_prompt = request.query.get("voice_prompt", "")
        runtime["voice_prompt"] = voice_prompt or None
        runtime["voice_loading"] = True
        runtime["handshake_ready"] = False
        runtime["active_chat_requests"] += 1
        _runtime_event(runtime, "personaplex.voice_prepare", voice_prompt=voice_prompt, remote=request.remote or "")
        try:
            response = await original_chat(request)
            runtime["handshake_ready"] = True
            _runtime_event(runtime, "personaplex.chat_ready", voice_prompt=voice_prompt)
            return response
        except Exception as exc:
            runtime["last_error"] = repr(exc)
            _runtime_event(runtime, "personaplex.chat_error", error=repr(exc))
            raise
        finally:
            runtime["voice_loading"] = False
            runtime["active_chat_requests"] = max(0, int(runtime["active_chat_requests"]) - 1)

    async def avatar_audio_with_runtime(request: web.Request):
        _runtime_event(runtime, "personaplex.mirror_subscribe", remote=request.remote or "")
        try:
            return await original_avatar_audio(request)
        except Exception as exc:
            runtime["last_error"] = repr(exc)
            _runtime_event(runtime, "personaplex.mirror_error", error=repr(exc))
            raise

    async def voices_with_runtime(request: web.Request):
        _runtime_event(runtime, "personaplex.voices_list")
        return await original_voices(request)

    async def personaplex_runtime_status(_request: web.Request):
        if cloudflare_provider is not None:
            runtime["cloudflare_turn_last_error"] = cloudflare_provider.last_error
        runtime["gpu"] = _cuda_mem_stats()
        return web.json_response(runtime)

    app.router.add_get("/api/chat", chat_with_runtime)
    app.router.add_get("/api/avatar/audio", avatar_audio_with_runtime)
    if original_voices is not None:
        app.router.add_get("/api/voices", voices_with_runtime)
    app.router.add_get("/personaplex/runtime", personaplex_runtime_status)


def _build_app() -> tuple[web.Application, str, int]:
    _patch_frontend()
    musetalk_args = parse_musetalk_args()
    print("[unified_server] Loading MuseTalk app...")
    _log_cuda_mem("musetalk.before_build_app")
    musetalk_app_state = WebRtcApp(musetalk_args)
    _install_cloudflare_turn(musetalk_app_state)
    if hasattr(musetalk_app_state, "cloudflare_turn_provider"):
        provider = musetalk_app_state.cloudflare_turn_provider
        try:
            cfg = provider.get_config()
            print(
                "[cloudflare-turn] startup probe succeeded "
                f"status={provider.last_status_code} servers={len(cfg.get('iceServers', []))} "
                f"identity={provider.debug_identity()}"
            )
        except Exception as exc:
            print(f"[cloudflare-turn] startup probe failed: {exc!r}")
    app = musetalk_app_state.build_app()
    _log_cuda_mem("musetalk.after_build_app")
    print("[unified_server] MuseTalk app ready.")
    if hasattr(musetalk_app_state, "cloudflare_turn_provider"):
        app["cloudflare_turn_provider"] = musetalk_app_state.cloudflare_turn_provider

    personaplex_state = _build_personaplex_state()
    app["personaplex_state"] = personaplex_state
    _install_personaplex_runtime_routes(app, personaplex_state)

    return app, musetalk_args.host, musetalk_args.port


def main() -> None:
    print("[unified_server] Starting unified PersonaPlex + MuseTalk server...")
    app, host, port = _build_app()
    cert_path = os.environ.get("ONEBOX_TLS_CERT_PATH", "/tmp/onebox.crt")
    key_path = os.environ.get("ONEBOX_TLS_KEY_PATH", "/tmp/onebox.key")
    internal_host = os.environ.get("ONEBOX_INTERNAL_HTTP_HOST", "127.0.0.1")
    internal_port = int(os.environ.get("ONEBOX_INTERNAL_HTTP_PORT", str(port + 1)))
    ssl_ctx = None
    scheme = "http"

    if os.path.exists(cert_path) and os.path.exists(key_path):
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(cert_path, key_path)
        scheme = "https"
        print(f"[unified_server] TLS enabled with cert={cert_path} key={key_path}")
    else:
        print(
            "[unified_server] TLS disabled because certificate files were not found: "
            f"cert={cert_path} key={key_path}"
        )

    print(f"[unified_server] Serving unified app on {scheme}://{host}:{port}")
    print(f"[unified_server] Serving internal HTTP loopback on http://{internal_host}:{internal_port}")

    async def _run() -> None:
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            public_site = web.TCPSite(runner, host=host, port=port, ssl_context=ssl_ctx)
            internal_site = web.TCPSite(runner, host=internal_host, port=internal_port)
            await public_site.start()
            await internal_site.start()
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()

    asyncio.run(_run())


if __name__ == "__main__":
    with torch.no_grad():
        main()
