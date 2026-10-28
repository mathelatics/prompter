# Setup 02 — Run Your First Session (every mode, copy-paste)

Verified on 2026-10-28 against `/workspace/cprompter` build outputs. The repo
ships `samples/demo.txt` (3 pages, 68 words: "Sponsor read", "Keynote intro", …).

## 1. Classic mode in the terminal (works everywhere, zero deps)

```sh
cd cprompter && make            # if not built yet
./cprompter samples/demo.txt --headless --speed 4
```
You get one ANSI line per highlighted word: dim = already spoken, **bold yellow**
= current, normal = upcoming; page headers shown. Ctrl-C to exit. This is also
what runs over SSH or on a Raspberry Pi framebuffer console.

```sh
make demo                       # same thing at 4 wps via Makefile shortcut
./cprompter samples/demo.txt --headless --loop   # endless rehearsal loop
```

## 2. Convert/inspect scripts (.textream round-trip)

```sh
./cprompter samples/demo.txt --save-out /tmp/mine.textream
# verified output: saved /tmp/mine.textream (3 pages, 68 words)
./cprompter /tmp/mine.textream --headless          # loads back with saved settings
```
`.textream` is JSON (`version`, `pages[] {id,name,content}`, per-page settings)
— safe to hand-edit or version-control. PowerPoint presenter notes: drop a
`.pptx` path as the script argument and notes are extracted slide-by-slide into
pages (needs `unzip` on PATH; see docs/proj/cprompter-04-formats-tools.md).

## 3. Voice mode (scrolls only while you speak)

```sh
make clean && make SDL=1        # overlay build; needs libsdl2
./cprompter /tmp/mine.textream --mode voice
```
- Opens mic via ALSA (dlopen'd; no link-time dep). No mic → warning + classic
  clock fallback, session still usable.
- Live waveform + measured WPM shown in the overlay.
- Hotkeys (shared table in `ui/hotkeys.h`): Space pause · ±/wheel speed ·
  ←→ scrub · [ ] prev/next page · **m cycles classic→voice→track** · r
  recalibrate VAD floor · +/- VAD sensitivity · s save .textream · q quit · h help.
- Headless CI trick without a mic: run tests/binaries with `CP_NULL_TONE=1` so
  the null audio device emits 1 kHz and VAD/WPM paths get exercised.

## 4. Word-tracking mode (word-by-word highlight from ASR)

```sh
# one-time model fetch (~150 MB tiny.en recommended first):
mkdir -p models
curl -L -o models/ggml-tiny.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
# code default path is models/ggml-base.en.bin — either download that name too
# or set model_path via the engine params when embedding.
make clean && make WHISPER=1    # after building third_party/whisper.cpp (setup-01 §4)
./cprompter /tmp/mine.textream --mode track
```
Speak naturally; the aligner advances the highlight to the exact word, tolerates
pauses, mild stutters, and re-reads ("the … the" backtrack rule is unit-tested).
Track-mode fallback chain: whisper → mock/file backend → voice → classic, so a
missing model never bricks a live show.

## 5. AI coach (M5) — rules offline, LLM optional

```sh
# Layer 1, fully offline heuristics (pace vs target, dead air, re-reads):
./cprompter samples/demo.txt --headless --coach rules --target-wpm 150
# prints cyan [coach] lines; auto-director can slow/speed/pause/advance pages.

# Layer 2, OpenAI-compatible advisor — LOCAL example (llama.cpp server):
CCP_BASE_URL=http://127.0.0.1:8080/v1 CCP_MODEL=qwen2.5-7b-instruct \
  ./cprompter samples/demo.txt --headless --coach llm

# CLOUD example (base URL + token, e.g. any provider):
CCP_BASE_URL=https://api.example.com/v1 CCP_MODEL=small-fast-model \
  CCP_API_KEY=sk-... \
  ./cprompter samples/demo.txt --headless --coach llm
```
Guarantees: bad URL → `coach: bad CCP_BASE_URL, falling back to rules`; LLM
latency/errors never block the pacer thread (advice arrives via mailbox);
OPENAI_API_KEY is accepted as fallback for CCP_API_KEY.

## 6. Remote viewer & Director Mode (SERVER=1)

The HTTP/WS layer (:7373 viewer, :7575 director, QR print helper) is complete,
links cleanly, and its route semantics (200/403/409, 10 Hz push) are covered by
engine-side tests. **CLI flags `--remote/--director` are the last unwired item**
(see setup-00 "Known debt" and cprompter-03-server.md §5 checklist). After that
wiring lands, usage will be:

```sh
./cprompter /tmp/mine.textream --remote --director   # planned UX
# scan printed QR -> phone browser opens read-only prompter at http://<lan-ip>:7373
# director edits/pushes at http://<lan-ip>:7575 ; delivered text returns HTTP 409
```

## 7. macOS / Windows quick notes

- macOS: `brew install sdl2` then identical commands; notch-style always-on-top
  window uses standard SDL window level flags (no private APIs needed there).
- Windows: build under WSL2 today; native MinGW cross-build is on the CI matrix
  plan (docs/proj/cprompter-06-testing-builds.md), not yet proven — don't promise
  users a .exe until it is.

Next: `setup-03-troubleshooting.md`.
