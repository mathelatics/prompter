# Setup 01 — Build Guide (all four flavors, copy-paste commands)

All commands below were **executed and verified** in `/workspace/cprompter` on
2026-10-28. Expected outputs are quoted from the real runs.

## TL;DR

```sh
cd cprompter
make test        # 93 checks, 0 failures   <- run this first, always
make             # headless binary -> ./cprompter  (~351 KB, zero deps)
make SDL=1       # + graphical overlay (needs libsdl2)
make SERVER=1    # + HTTP/WS remote & director code linked in
make WHISPER=1   # + word-tracking ASR (needs third_party/whisper.cpp built)
```

## Flavor 1 — default (headless core, zero dependencies)

```sh
cd cprompter && make
```
Compiles `main.c` + `core/ format/ audio/ ai/ ui/hotkeys.c` with
`-O2 -std=c11 -Wall -Wextra -g`, links `-lm -lpthread -ldl`.
Verified result: clean build, no warnings, `./cprompter --help` works.

## Flavor 2 — SDL overlay

```sh
sudo apt install libsdl2-dev            # or: brew install sdl2
# optional nicer text: sudo apt install libsdl2-ttf-dev
cd cprompter && make clean && make SDL=1
```
- Makefile auto-detects flags via `pkg-config --cflags/--libs sdl2`
  (falls back to `-I/usr/include/SDL2 -lSDL2`).
- If `pkg-config --exists SDL2_ttf` succeeds it also defines
  `CP_HAVE_SDL_TTF=1` automatically (TrueType font rendering).
- Overlay only activates when stdout is a TTY and `--headless` was NOT passed;
  otherwise you still get the ANSI terminal renderer (great over SSH).

## Flavor 3 — SERVER=1 (M3 remote viewer + director)

```sh
cd cprompter && make clean && make SERVER=1
```
Adds `server/http_ws.c server/remote.c vendor/qrcodegen.c` (pure POSIX sockets,
no extra libs). Verified: compiles warning-free, links into `./cprompter`;
`nm cprompter | grep -c remote_start` → `1`.

⚠️ Runtime caveat (tracked debt): `main.c` does not yet expose `--remote` /
`--director` flags that call `cp_remote_start/tick/stop`. The server code is
complete and unit-exercised; CLI wiring is the remaining checklist item in
`docs/proj/cprompter-03-server.md` §5. Until wired, exercise routes through a
harness linking `server/*.o`.

## Flavor 4 — WHISPER=1 (M2 word tracking)

```sh
mkdir -p third_party && cd third_party
git clone --depth 1 https://github.com/ggerganov/whisper.cpp
cd whisper.cpp && cmake -B build && cmake --build build -j
cd ../.. && make clean && make WHISPER=1
```
Makefile links `-Lthird_party/whisper.cpp/build/src -lwhisper -lggml-base` and
auto-adds `build/src/libggml-cpu.a` if present. Define `CP_HAVE_WHISPER=1` so
`audio/track.c` compiles its whisper backend; without the flag the file builds
a graceful "track backend unavailable" path instead.

Combining flags works: `make SDL=1 SERVER=1 WHISPER=1 CURL=1`.

## Optional — curl for https cloud endpoints

```sh
make CURL=1 ...    # defines CP_HAVE_CURL=1 (link -lcurl as needed by your toolchain)
```
Without it, `ai/http_client.c` speaks plain HTTP over POSIX sockets — sufficient
for local llama.cpp/Ollama (`http://127.0.0.1:.../v1`).

## Tests

```sh
cd cprompter && make test
```
Builds `cp_tests` from `tools/test_core.c` + all core modules with
`-DCP_TESTING=1`, then runs it. Verified output:

```
93 checks, 0 failures
```

Covers: script model, pacer semantics (classic/voice/track), .textream
round-trip, pptx XML extraction, VAD/WPM, aligner/fuzzy-match ladder, track
mock pipeline (real wall-clock loop), read-lock edit conflicts, coach rules,
JSON parser, hotkey map.

## Gotchas (learned the hard way — see docs/proj/cprompter-06-testing-builds.md)

1. **Stale objects across flavors**: `.o` files don't record which flags built
   them. ALWAYS `make clean` before switching `SDL/WHISPER/SERVER/CURL` values.
2. `make demo` = `./cprompter samples/demo.txt --speed 4` (terminal renderer).
3. On minimal containers voice mode prints an ALSA-open failure and falls back
   to classic clock — that's by design, not a broken build.
4. `models/ggml-base.en.bin` must exist relative to your CWD for WHISPER runs
   (see setup-02 for download command).

Next: `setup-02-run-first-session.md`.
