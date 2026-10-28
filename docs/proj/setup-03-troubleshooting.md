# Setup 03 — Troubleshooting & FAQ

Every symptom below was matched to real strings/behaviors in the codebase (file:
line references included) so fixes target actual code paths, not folklore.

## Build problems

| Symptom | Cause | Fix |
|---|---|---|
| `undefined reference to cp_pacer_set_speed` | stale `.o` from before that function landed | `make clean && make` (objects don't track source-history; always clean after pulls) |
| `make SDL=1` fails with `SDL.h: No such file` | libsdl2-dev missing or pkg-config absent | install it (`apt install libsdl2-dev` / `brew install sdl2`); Makefile fallback assumes headers under `/usr/include/SDL2` |
| `make WHISPER=1`: `-lwhisper` not found | whisper.cpp not built, wrong dir | need `third_party/whisper.cpp/build/src/libwhisper.a`; re-run cmake steps from setup-01 §4; override root with `make WHISPER=1 WROOT=/path/to/whisper.cpp` |
| Link error on `libggml-cpu.a` | newer whisper.cpp splits ggml | ensure ggml-base built (default), pass extra libs via `LDLIBS += ...` on your make line |
| Weird crash only in one flavor | mixed objects built with different flags (CP_HAVE_SDL vs not) | `make clean` between every flag change — the #1 gotcha |
| `usleep` implicit-decl on strict libc | needs `_DEFAULT_SOURCE` | already defined in main.c; add same define at top of any new TU using usleep |

## Runtime problems

### "engine: libasound not found (falling back to classic clock)"
`audio/audio.c:111`. Expected on containers/macOS-without-ALSA. Voice mode can't
capture; classic timing still works. Linux: `sudo apt install libasound2`. macOS
uses CoreAudio backend path when present; Windows: WSL2 for now.

### Mic opens but VAD never triggers
- Speak within ~30 cm; check input isn't muted (`alsamixer`).
- Press **r** to recalibrate the noise floor after your room/device settles.
- Lower threshold with **+** sensitivity hotkey (VAD hangover documented in
  docs/proj/cprompter-02-audio-ui.md).
- Verify capture itself: run tests with `CP_NULL_TONE=1` — if waveform moves,
  pipeline is fine and the issue is level/threshold.

### Track mode: highlight stuck at word 0
- Model file missing: default lookup is `models/ggml-base.en.bin` relative to
  CWD (`audio/track.c:152`). Download per setup-02 §4 or set `model_path`.
- Whisper init failure is non-fatal — engine falls back voice→classic; watch
  stderr for the reason string.
- Heavy CPU (RPi + base.en): use `tiny.en`; latency budget <100 ms needs ≥4 cores.

### Coach does nothing / spam errors
- `--coach rules` only reacts when measured WPM exists → must be voice/track
  session OR use `CP_NULL_TONE=1` demo runs.
- LLM layer: bad URL prints `coach: bad CCP_BASE_URL, falling back to rules`
  (main.c:178) and continues offline. For https endpoints without CURL=1 build,
  switch to a local http:// proxy (llama.cpp `--api`, Ollama, LiteLLM).
- Advice appears on stderr `[coach] …`; stdout pipe redirection may hide it —
  check `2>&1`.

### Remote/Director pages unreachable
1. First confirm you're on a SERVER=1 binary: `nm ./cprompter | grep remote_start`.
2. Known debt: `--remote/--director` CLI wiring in main.c is NOT yet done
   (docs/proj/cprompter-03-server.md §5). Until wired, no port binds — this is
   expected, not a firewall bug.
3. After wiring: viewer :7373 read-only (POSTs → 403), director :7575; edits
   touching delivered words → HTTP 409 by design (read-lock). Ports are args to
   `cp_remote_start()` — change them there if 7373/7575 clash.

### Overlay invisible / behind other windows
Window level + always-on-top flags live in `ui/overlay_sdl.c`; some Wayland
compositors ignore `_NET_WM_STATE_ABOVE`. Workarounds: fullscreen mode, or
X11 session. Follow-mouse/fixed-display modes are selectable at runtime
(hotkey table in docs/proj/cprompter-02-audio-ui.md).

## Data problems

| Symptom | Fix |
|---|---|
| `.textream` won't load | validate JSON: keys `version`, `pages[]` with `content`; hand-corruption possible since format is plain JSON — regenerate via `--save-out` |
| pptx import empty | notes live in `notesSlides/`; decks created without presenter notes have nothing to extract; needs `unzip` binary on PATH |
| Curly quotes / em-dashes mis-highlighted | normalization ladder strips them before matching (`audio/align.c` `cp_norm_word`) — update aligner tests rather than pre-editing scripts |

## FAQ

**Does anything leave my machine?** No, except two opt-ins: an LLM coach you
explicitly point at a cloud `CCP_BASE_URL`, and LAN browsers connecting to
:7373/:7575. Speech recognition (whisper.cpp) is fully local.

**Can I script/embed it?** Yes — link `core/ audio/ ai/ server/` objects; each
module has a header contract and unit tests. Threading rule: pacer tick stays on
your UI thread; audio + coach own their threads and communicate via callbacks /
mailbox (`cp_coach_poll`).

**Why do tests print "93 checks"?** That's `tools/test_core.c` total assertions
(verified current). If you add modules, extend it and update the count in
setup-01/docs — CI greps it.

**Where is the roadmap?** M6+ items and open debt tracked in
`docs/proj/cprompter-04-formats-tools.md` ("not done" list) and
`docs/audit/cprompter-03-known-issues.md`.
