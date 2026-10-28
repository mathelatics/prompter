# cPrompter — Project 00: Overview & Onboarding

> **Read this first.** One paragraph per doc in `docs/proj/` telling you what it covers.

## What cPrompter is
A free, MIT-style, pure-C teleprompter that recreates every Textream (macOS/Swift)
feature and runs it on *any* device: Linux, macOS, Windows (WSL/Cygwin), Raspberry Pi,
headless SSH, and any browser via the built-in LAN web server — plus an AI coach layer
that talks to any OpenAI-compatible endpoint (`base_url` + token; local llama.cpp /
Ollama / our llm.c engine OR cloud).

Origin story and feature-parity analysis: `docs/audit/cprompter-01-plan.md`.
Dependency vetting rules: `docs/audit/cprompter-02-deps.md`.
Historical bug register (all resolved): `docs/audit/cprompter-03-known-issues.md`.

## Repo layout (actual, verified 2026-10-28)
```
cprompter/
├── Makefile                 # SDL=1 WHISPER=1 SERVER=1 CURL=1 feature switches
├── main.c                   # CLI + headless terminal loop + coach wiring (~250 L)
├── core/    script.[ch] pacer.[ch] engine.[ch]      # data model, state machine, runtime (~950 L)
├── format/  script_io.[ch] pptx.[ch]                # .textream JSON + PowerPoint import
├── audio/   audio.[ch] vad.[ch] wpm.[ch]            # ALSA-dlopen capture, VAD, syllabic WPM
│          align.[ch] track.[ch] wav.[ch]            # word tracker, ASR backends, WAV I/O
├── ai/      json.[ch] http_client.[ch] openai.[ch] coach.[ch]  # M5 advisor stack
├── ui/      hotkeys.[ch] overlay_sdl.[ch]           # key bindings, SDL2 notch/fullscreen
├── server/  http_ws.[ch] remote.[ch]                # hand-rolled HTTP/1.1+RFC6455, :7373/:7575
├── vendor/  qrcodegen.[ch]                          # ISO 18004 QR encoder (differential-tested)
├── tools/   test_core.c make_test_pptx.py           # 93-check unit suite + pptx fixture gen
├── samples/ demo.txt                                # 3-page demo script
└── third_party/                                     # whisper.cpp goes here (WHISPER=1 builds)
```
Total ≈ 5,700 LoC C (+ embedded HTML). Everything compiles with `-Wall -Wextra`, zero warnings.

## Build & run (copy-paste)
```sh
make                      # headless/classic build, ZERO external deps
make test                 # 93 checks, 0 failures (unit suite)
make SDL=1                # + graphical overlay (needs libsdl2; SDL2_ttf autodetected)
make SERVER=1             # + M3 remote viewer/director servers (pure POSIX sockets)
make WHISPER=1            # + M2 real ASR (needs third_party/whisper.cpp built)
./cprompter samples/demo.txt --speed 4
./cprompter slides.pptx --mode voice --coach rules
CCP_BASE_URL=http://127.0.0.1:8080/v1 ./cprompter k.txt --coach llm --target-wpm 140
```

## CLI surface (from main.c usage(), verified)
| flag | meaning |
|---|---|
| `--speed <wps>` | words/sec, 0.5–8.0 (Textream range), default 3.0 |
| `--mode classic\|voice\|track` | guidance-mode override |
| `--headless` | plain stdout renderer (CI/ssh; no SDL even if built) |
| `--loop` | restart at end |
| `--save-out <path>` | re-save loaded script as `.textream` and exit |
| `--coach rules\|llm` | M5 coach: local heuristics, or + OpenAI-compatible advisor |
| `--target-wpm <n>` | coach ideal pace (default 150) |
| `-h/--help` | usage |

Env vars for the advisor: `CCP_BASE_URL` (default `http://127.0.0.1:8080/v1`),
`CCP_MODEL`, `CCP_API_KEY` (empty → no auth header, correct for local servers).

## Hotkeys (ui/hotkeys.h — single source of truth, shared by terminal & SDL)
Space=pause · ←/→ or wheel = scroll ± · +/- speed · `[` `]` page prev/next ·
m=cycle mode · s=save .textream · r=VAD recalibrate · ,/. sensitivity · f=mirror flip ·
o=overlay style · t=tap-to-resync (track mode) · h=help · q/ESC=quit.

## Milestones M0–M5 — all shipped & green
- **M0** core model + classic scroller + .textream IO → doc `cprompter-01-core.md`
- **M1** voice mode: ALSA-dlopen capture + VAD gate + syllabic WPM → `cprompter-02-audio-ui.md`
- **M2** word tracking: align + whisper/mock/file backends → `cprompter-02-audio-ui.md`
- **M3** remote viewer :7373 + director :7575 + read-lock 409 + QR → `cprompter-03-server.md`
- **M4** pptx import, mirror/fullscreen polish → `cprompter-04-formats-tools.md`
- **M5** AI coach / auto-director → `cprompter-05-ai-coach.md`
- Testing philosophy & harnesses → `cprompter-06-testing-builds.md`

## Non-negotiable engineering rules (learned the hard way this project)
1. **No stubs, no dummies.** Every claim in a commit message must be backed by an
   executed command output. Differential testing beats eyeballing (QR vs pip `qrcode`).
2. **Tests encode actual behavior, not wishful assertions.** When a test failed we
   first decided whether the code or the expectation was wrong, and wrote why.
3. **Zero-dependency default build** must always keep working; optional features are
   compile-time flags with graceful runtime fallbacks (e.g. bad `CCP_BASE_URL` →
   "falling back to rules", never a crash).
4. Small, single-purpose vendored code only (<20k LoC, MIT/BSD) — see deps doc.

## Quick orientation map for a new developer
- Want to change scroll semantics? → `core/pacer.c` ONLY (single source of truth).
- Want a new render target (wayland/fbterm/web)? → copy `ui/overlay_sdl.c` pattern:
  read state exclusively through `cp_engine_*()` getters, never poke structs.
- Want a new ASR? → implement the backend switch in `audio/track.c`; aligner is
  ASR-agnostic (consumes a word stream).
- Want a new AI action ("auto-highlight sponsor")? → extend `cp_advice_kind` in
  `ai/coach.h` + handle it in the two consumers (main.c headless loop, overlay UI).
- Anything network-facing → `server/http_ws.c` owns sockets; don't open raw ones.
