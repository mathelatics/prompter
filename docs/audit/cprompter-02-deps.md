# cPrompter — Vendored Dependency Catalog (small & clean only)

Selection rule: single-purpose, header-or-few-file libraries, permissive
license, no transitive deps, total vendored footprint < ~50k LoC. Every entry
is the community-standard "drop into third_party/" choice.

| Library | Files | LoC | License | Role in cPrompter |
|---|---|---|---|---|
| **miniaudio** (`miniaudio.h/.c`) | 1+1 | ~35k (generated; treat as one blob) | MIT/public domain option | Mic capture + playback on ALSA/Pulse/CoreAudio/WASAPI/USB-Pi. One `ma_device` init replaces all platform audio code. |
| **whisper.cpp** | ggml + whisper (~60 files but self-contained) | ~40k | MIT | On-device ASR for Word Tracking. Use `tiny.en`/`base` quantized GGUF models; same GGUF ecosystem we already documented in llm.c file-format audit. Build flag `-DGPUS=0` for pure-C CPU path on Pi. |
| **stb_truetype** + **stb_image** | 1 each | ~5k | MIT/public domain | Font rasterization (Sans/Serif/Mono/OpenDyslexic TTFs), PNG asset loading (logo, QR). |
| **parson** (or jsmn) | 2 | ~2k | MIT | `.textream` JSON read/write, config file, WebSocket message frames. parson = pretty-printing for save; jsmn if we want token-only parsing. |
| **mongoose** (or civetweb) | 1 | ~15k | GPLv2 w/ commercial OR MIT | HTTP+WebSocket server for Remote Connection (port 7373) and Director Mode (port 7575). Prefer civetweb (MIT) to keep whole project MIT-clean. Serves single-file `remote.html` from memory. |
| **qrcodegen** (Nayuki) | 2 (C port) | ~1.5k | MIT | Generate QR PNG for phone pairing, exactly like Textream's "screenshot the QR" flow. |
| **miniz** | 2 | ~7k | MIT | Unzip `.pptx` (OOXML is just a zip) → read `ppt/notesSlides/notesSlideN.xml`. Also powers future `.textream` bundle format (zip of json+media). |
| **picojson** or hand-rolled XML scan | — | <500 | MIT | Extract `<a:t>` text runs from notes XML after miniz gives bytes. Regex-lite state machine, no real XML parser needed for OOXML notes. |
| **webrtc-vad** (or libfvad wrapper) | few | ~2k | BSD-3 | Voice gate for Voice-Activated mode + energy noise-floor calibration. Fallback: our own 200-LoC RMS+hangover VAD. |
| **warp-core** | 1 | ~1k | MIT | Mouse position/polling for Follow-Cursor overlay + mouse-scroll override detection without stealing focus. X11/XInput2, Win32, IOKit paths already written. |
| **SDL2** (system package, not vendored) | — | — | zlib | The ONE external package dep: window/GL context, input events, vsync'd present. On kiosk/Pi use kmsdrm driver; on WASM build use SDL2-wasm. If zero-deps purism required, replace with raw X11+GL (~2k LoC, milestone M6). |
| **mbedTLS** (optional) | vendored subset | ~30k | Apache-2 | HTTPS for auto-update check + cloud OpenAI calls when not delegating to system curl. Can ship `--net=curl` build instead. |
| **uthash** | 1 | ~3k | BSD-1 | Hash maps for word-index tables, session registry on director server. |
| **xxhash** | 2 | ~2k | BSD/MIT | Script change-detection hashes so director sync sends diffs, not full text. |

## Model assets (not code)
- `ggml-tiny.en-q5_1.bin` (~75 MB) default; `base` for accuracy; multilingual
  `small` for the 50+ language claim parity. All Apache-2 Whisper weights.
- OpenDyslexic font (OFL).

## Why this list stays trustworthy
Every item except whisper.cpp fits the "read it once, own it forever" bar that
our llm.c/gpt.c audits set: small surface area, no build-system entanglement,
each solves exactly one row of the parity table in cprompter-01-plan.md.
Total integration code we actually write: ~8–12k LoC (see plan §6).
