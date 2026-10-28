# Setup 00 — Prerequisites & System Requirements

Everything needed before building cPrompter. Verified against the actual tree in
`/workspace/cprompter` (Makefile, sources) on 2026-10-28.

## Hardware / OS baseline

| Requirement | Minimum | Notes |
|---|---|---|
| CPU | any x86_64 / ARM64 | pure C11, no SIMD required |
| RAM | 64 MB (headless) | +~300 MB if whisper `base.en`; ~150 MB for `tiny.en` |
| OS | Linux (glibc), macOS 12+, Windows via WSL2/MinGW | built & tested on Linux; POSIX sockets/threads only |
| Compiler | GCC 9+ or Clang 12+ | `-std=c11 -Wall -Wextra`, builds warning-free |
| make | GNU make | Makefile uses `$(filter)`/`ifeq` conditionals |

## What is mandatory vs optional

**Mandatory for every build: nothing beyond a C compiler + libc/libm/libpthread/libdl.**
The default `make` target has **zero external library dependencies**:

- ALSA capture is loaded at runtime via `dlopen("libasound.so.2")`
  (`audio/audio.c`) — voice mode works if present, degrades gracefully if not.
- HTTP/WebSocket server (`server/http_ws.c`) is hand-rolled over POSIX sockets —
  no civetweb needed.
- QR encoder (`vendor/qrcodegen.c`) is vendored in-tree — no dependency.
- JSON parser (`ai/json.c`) and HTTP client (`ai/http_client.c`, plain POSIX
  sockets for `http://` endpoints) are in-tree.

**Optional components** (each auto-detected or flag-gated):

| Component | Needed for | How obtained |
|---|---|---|
| `libsdl2-dev` (+ optionally `libsdl2-ttf-dev`) | graphical overlay (`make SDL=1`) | `apt install libsdl2-dev libsdl2-ttf-dev` / `brew install sdl2 sdl2_ttf` |
| `whisper.cpp` | word-tracking ASR backend (`make WHISPER=1`) | clone into `third_party/whisper.cpp`, cmake-build (see setup-02) |
| `curl` (optional, `make CURL=1`) | `https://` cloud LLM endpoints | system package; else use local `http://` base URLs |
| `python3` + `pip` | QR differential test harness only | not needed to run the app |
| microphone | voice/track modes | USB condenser recommended; laptop mic fine |
| any OpenAI-compatible server | M5 LLM coach | llama.cpp/Ollama/LM Studio local, or cloud w/ token |

## Environment variables consumed by the binary (complete list)

Verified by grep of the sources:

| Variable | Read in | Effect |
|---|---|---|
| `CCP_BASE_URL` | `ai/openai.c:30` | OpenAI-compatible endpoint, e.g. `http://127.0.0.1:8080/v1` |
| `CCP_MODEL` | `ai/openai.c:33` | model name for the coach chat-completions call |
| `CCP_API_KEY` | `ai/openai.c:35` | bearer token (cloud); leave empty for local servers |
| `OPENAI_API_KEY` | `ai/openai.c:36` | fallback for `CCP_API_KEY` |
| `CP_NULL_TONE` | `audio/audio.c:142` | `=1` makes the null CI audio device emit a 1 kHz tone (drives VAD/WPM tests without a mic) |

Bad `CCP_BASE_URL` → prints `coach: bad CCP_BASE_URL, falling back to rules` and
runs the offline rules coach instead (no crash).

## Disk layout expectations

- Whisper models are looked up at **`models/ggml-base.en.bin` relative to the
  working directory** unless overridden (`audio/track.c:152`, `model_path` in
  `cp_track_params`). Create `models/` next to where you run the binary.
- `.textream` scripts and pasted text are plain files anywhere.
- `third_party/` must contain `whisper.cpp/` with a `build/` dir when using
  `WHISPER=1` (Makefile links `-L$(WROOT)/build/src -lwhisper -lggml-base` and
  adds `libggml-cpu.a` if present).

## Network ports (only used when running with server support)

| Port | Purpose | Configurable? |
|---|---|---|
| 7373 | Remote Viewer (read-only web page + WS, matches Textream default) | yes — `cp_remote_start(eng, viewer_port, director_port)` takes ports as args (`server/remote.h:22`) |
| 7575 | Director Mode (editor web UI + WS, matches Textream default) | same |

Both bind local interfaces only; traffic never leaves your LAN by design.

## Known debt affecting setup (from docs/proj/cprompter-03-server.md §5)

The SERVER=1 binary compiles and links cleanly today (verified: `nm cprompter |
grep remote_start` finds it), but **main.c does not yet parse `--remote` /
`--director` flags nor call `cp_remote_start/tick/stop`** — until that wiring
lands, run integration checks via the test harnesses rather than expecting a
live server from the CLI. This is documented honestly so nobody hunts for a
missing flag.

Next: `setup-01-build.md`.
