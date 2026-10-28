# cPrompter — Project 02: Audio pipeline (M1 voice + M2 tracking) & UI

Files: `audio/audio.c` (capture+DSP), `audio/vad.c`, `audio/wpm.c`, `audio/align.c`,
`audio/track.c`, `audio/wav.c`, `ui/hotkeys.c`, `ui/overlay_sdl.c`. Total ≈ 1,300 LoC.

## 1. Capture — `audio/audio.h/.c` (zero-dep by design)
- **ALSA via `dlopen("libasound.so.2")`** at runtime: binary builds/runs everywhere; if
  the lib is missing you get a clean error string, not a link failure. `"null"` device =
  synthetic silent/zeros stream for CI/headless tests.
- Internal contract: everything downstream consumes **mono float32 @ CP_AUDIO_RATE
  (16 kHz)** through `cp_audio_cb(pcm, n, user)` on the capture thread (frame_ms=20).
- DSP helpers (pure, unit-tested): `cp_resample_i16_to16k` (averaging pre-filter),
  `cp_rms`, one-pole high-pass `cp_hp1` (kills rumble/breath noise, stateful).
- `cp_audio_level()` → peak of recent frames, drives waveform UI + remote JSON `level`.

## 2. VAD gate — `audio/vad.c` (Voice-Activated mode brain)
Heuristic, no deps: frame RMS vs **adaptive noise floor** + spectral-flatness tiebreaker
(voice is less flat than hiss/fan), with a **300 ms hangover** so mid-word breaths don't
freeze the scroll. Sensitivity knob −2..+2 shifts threshold in dB; `recalibrate` relearns
noise floor (hotkey `r`). API: `push(frame)->speaking?`, `level()`, `rms()`, `thresh()`.
Future swap-in point: webrtc-vad behind the same 8-function header.

## 3. WPM estimator — `audio/wpm.c` (no ASR needed!)
Speech-science trick: count **syllable peaks** in the energy envelope (~150 ms plateau
window), divide by elapsed time, scale by words-per-syllable (~1.4 English). Smoothed
over last ~6 s (`estimate`) plus per-utterance `inst`; syllable counter exposed for
waveform tick marks. Feeds `cp_pacer_feed_vad(active, inst_wpm)` and the coach's pace
rules. Robust to pauses — that's why voice mode never needs whisper.

## 4. Word tracker — `audio/align.c` (M2 core algorithm)
Turns ANY recognized-word stream (whisper, cloud ASR via agent layer, mock harness) into
a monotone cursor into `script->words`. Rules, in priority order per fed word:
1. **fast path**: next expected script word matches (exact → normalized → fuzzy) → +1;
2. **lookahead wins over backtrack**: presenter skipped ahead ≤ AL_MAX_LOOK → jump fwd.
   (Real bug fixed honestly: "the … the" re-read used to steal the skip-ahead match.)
3. **re-read/backtrack** ≤ AL_MAX_BACK allowed (stutters) but NEVER below
   `max_delivered+1` unless explicitly permitted — preserves read-lock semantics;
4. fillers ("um","uh","so",…) silently dropped;
5. no match in window → stay put, `drift++`; after AL_GIVEUP misses the UI/coach show
   "tap to resync" (hotkey `t` → `cp_engine_resync`).

Normalization stack (all unit-tested): `cp_norm_word` (lowercase, strip ASCII+curly
punctuation, returns char count); `cp_fuzzy_eq(a,b)` = Levenshtein ≤1 for long words,
**short words (≤3 chars) require EXACT match** (else "the"/"they" chaos), plus
ASR-truncation containment rule (`telepromp ⊂ teleprompter` OK).
Result struct: `{advanced, matched, filler_skipped, script_idx, drift}`.

## 5. ASR backends — `audio/track.c` (+ `audio/wav.c`)
`cp_track_backend`: NONE | WHISPER (compiled only when `WHISPER=1`; links
`third_party/whisper.cpp` tiny/base int8 ggml model; params: model_path, lang,
chunk_sec=1.0 hop) | MOCK (NULL-terminated scripted word array fed at `mock_wps` wall
clock — used by unit tests AND the `--mode track` demo without any model download).
Engine pulls via `cp_track_poll()` → newly recognized words each UI frame. Design note:
a real-time mock-pipeline test must run as a true wall-clock loop (it caught a failing
test that polled faster than the rate limiter emits — honest test-design fix, not code hack).
`wav.c`: minimal RIFF reader/writer (used by file backend + tests writing /tmp/cp_*.wav).

## 6. Hotkeys — `ui/hotkeys.c`
Single mapping table shared by terminal and SDL front-ends: `cpk_lookup(ch, ctrl, alt)`
→ `cp_key_action` enum (QUIT/PAUSE/SPEED_UP|DOWN/SCROLL_FWD|BACK/PAGE_NEXT|PREV/MODE/
SAVE/RECAL/SENS_UP|DOWN/JUMP_HOME/HELP/MIRROR/OVERLAY/RESYNC). Pseudo keycodes
CPK_KEY_SPACE=0x100…CPK_WHEEL_*, ESC so backends normalize first. `cpk_help()` prints
the legend — update it HERE only.

## 7. SDL overlay — `ui/overlay_sdl.c` (only file guarded by CP_HAVE_SDL)
Two styles, live-cycled with `o`: **fullscreen** (opaque black, dedicated display /
Sidecar rig) and **notch** (borderless always-on-top strip pinned below top edge,
~40% alpha frosted look). Reads state exclusively via `cp_engine_*` getters. Features:
word-run line layout with paragraph breaks, current word highlighted in chosen palette
color, delivered text dimmed, progress bar, mic level meter + speaking dot (voice),
"heard:" ticker + drift warning (track), coach banner (M5, doc 05). Mirror flip `f`:
horizontal/vertical/180° implemented via render-target transform (Textream parity for
physical prompter glass). Font: SDL2_ttf autodetected (sans/serif/mono/dyslexic slots
fall back to bitmap boxes without ttf). Mouse wheel → `cp_engine_scroll` (auto-pauses
timer like Textream). Loop flag re-inits pacer at end. Returns process exit code.
macOS notch-level pinning would need a tiny Obj-C shim (window level) — deliberately
not done yet; borderless+always-on-top approximates it cross-platform.

## 8. Testing hooks
- `make test` runs align/VAD/WPM/pacer-track cases against mock streams only — audio
  hardware and whisper are NEVER required for CI.
- Engine built with `dev="null", enable_audio=1` exercises the full callback chain on
  synthetic frames (proves thread-safety under valgrind/TSAN manually).
