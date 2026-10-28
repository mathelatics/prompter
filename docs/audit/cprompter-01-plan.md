# cPrompter ("Textream in C") — Master Plan

Goal: rebuild everything Textream does for macOS as a **pure-C, dependency-light,
cross-platform teleprompter + AI presentation agent** that runs on Linux, macOS,
Windows (MinGW), Raspberry Pi, and in a browser via WASM — with an optional
OpenAI-compatible-model brain (local llama.cpp/Ollama/our llm.c engine OR cloud
via base_url+token) for smart pacing, ad-lib detection, and director automation.

Reference product: Textream (github.com/f/textream, MIT, Swift/macOS-only).
Our edge: same UX class, but any device, tiny footprint, scriptable, hackable,
and AI-augmented in ways the Swift app isn't.

---

## 1. Feature parity map (Textream → C module)

| Textream feature | C module | Key library (small/clean) |
|---|---|---|
| Word-tracking highlight (on-device ASR) | `asr/` | whisper.cpp (single .cpp+.h pair, ggml — same lineage as our llm.c docs) |
| Classic constant-speed scroll | `core/pacer.c` | none (pure math) |
| Voice-activated scroll (mic level gating) | `asr/vad.c` | webrtc-vad (~2k LoC) or energy+noise-gate in ~200 LoC |
| Notch/Dynamic-Island overlay | `ui/overlay_*.c` | SDL2 (borderless always-on-top window) — the only "big" dep, still tiny |
| Follow-mouse / fixed-display / floating window | `ui/overlay_*.c` | X11/EWMH `_NET_WM_STATE_ABOVE`, macOS `NSWindow level` shim, Win32 `HWND_TOPMOST` |
| Fullscreen + mirror flip (H/V/180°) | `render/flip.c` | shader flag or CPU blit transpose |
| Remote Connection (browser view, QR, 10 Hz WS) | `net/remote.c` | Mongoose (1 file, ~15k LoC) or civetweb; qrcodegen (2 files) |
| Director Mode (edit+push scripts from phone) | `net/director.c` | same HTTP/WS server + JSON |
| .pptx presenter-notes import | `import/pptx.c` | miniz (zip, 2 files) + a 300-LoC XML walker over `ppt/notesSlides/*.xml` |
| .textream JSON save format | `format/script.c` | parson or jsmn (tiny JSON) |
| Multi-page + auto-advance countdown | `core/pages.c` | none |
| Mouse-scroll override w/ timer pause | `input/warp.c` | warp-core (X11/win32 cursor pos, no libs needed beyond OS calls) |
| 50+ speech languages | free via whisper.cpp multilingual models | |
| Invisible-to-screen-recording overlay | platform-limited; see §6 caveats | |
| Auto-update check vs GitHub Releases | `net/update.c` | raw HTTPS via mbedTLS (optional; can shell to curl) |

All libraries above are single-purpose, MIT/BSD/Apache-licensed, each under
~20k lines, vendored into `third_party/`. No framework, no Electron, no Python.

---

## 2. Architecture (one binary, five layers)

```
main.c
 └─ core/          state machine: Script -> Words[] -> Cursor -> Pace
 │   pacer.c       WPM clock + drift correction (PID-style smoothing)
 │   align.c       word alignment: ASR tokens <-> script tokens (edit distance)
 │   pages.c       multi-page, countdown, read-lock (director)
 ├─ asr/           whisper.cpp wrapper + VAD gate; streaming chunks of 1s
 ├─ render/        SDL2 renderer: frosted panel, highlight colors, fonts
 │                 (stb_truetype for OpenDyslexic/Serif/Mono loading)
 ├─ net/           mongoose-based HTTP+WS server
 │   remote.html   zero-build single-file SPA served from RAM string
 │   director      REST: POST /script, WS /sync (state at 10Hz like Textream)
 └─ ai/            OpenAI-compatible client (base_url + token, see agent-c-02 doc)
     coach         live feedback: pace too fast, filler words, drift >5s
     aut-director  rewrites/promotes next paragraph mid-show via Director API
```

Core data model (mirrors `.textream` JSON from the blog post):
```c
typedef struct { char *text; int page; double t_est; float conf; } Word;
typedef struct { Word *ws; int n; PageCfg cfg; } Page;   // guidanceMode,
typedef struct { Page *ps; int np; Meta m; } Script;     // scrollSpeed, lang
```

Three guidance modes = one loop, three cursor-update sources:
- CLASSIC: `cursor += speed * dt`
- VOICE:   `if vad_active(): cursor += max(speed_min, measured_wpm)`
- TRACK:   `cursor = align(asr_partial_words, script_words)` (the hard part, §4)

---

## 3. Build & portability matrix

| Target | Overlay trick | Audio | Notes |
|---|---|---|---|
| Linux/X11 | borderless SDL win + `_NET_WM_STATE_ABOVE`, skip-taskbar, sticky-all-workspaces | ALSA/Pulse via miniaudio (1 header) | notch position = top-center offset; works on Wayland too via layer-shell later |
| macOS | same SDL path; optional tiny Obj-C shim (`overlay_mac.m`, ~120 LoC) for real NSWindow `.floating` level + `ignoresMouseEvents` | CoreAudio via miniaudio | runs on Intel + Silicon; no notarization pain if built locally |
| Windows | `SetWindowPos(HWND_TOPTOPMOST)` + layered window alpha | WASAPI via miniaudio | |
| Raspberry Pi / kiosk | fullscreen DRI/DRM via SDL, HDMI out as physical teleprompter rig | USB mic | this is where C shines — Textream can't go here |
| Android NDK | SDL2-android termux-x11 or native surface | SL4A mic | stretch |
| WASM (browser) | pure classic mode + Web Speech API instead of whisper | getUserMedia | demo funnel: try it on website, install binary for full features |

Single Makefile + CMake fallback. `make cpu` builds without whisper (classic
mode only) so even a $5 router-class box can run the prompter.

---

## 4. The hard problems (and how we actually solve them)

1. **Word tracking accuracy.** Whisper.cpp gives segment timestamps; we do
   forced-alignment lite: fuzzy-match ASR words to script words with
   Needleman-Wunsch over normalized tokens (lowercase, strip punctuation).
   Handle skips/restarts: allow cursor to jump backward only on explicit
   keypress; forward jumps allowed when match score > threshold. This is the
   same edit-distance machinery already documented in our audit style — keep
   it <500 LoC.
2. **Latency.** Streaming: 1s audio chunks, tiny.en/tiny-whisper model
   (~75 MB, 40x realtime on M-series, ~5x on Pi4 with NEON). Sub-300ms
   practical word confirmation — comparable to Textream's claimed <100 ms for
   Apple SFSpeech because Apple uses phoneme-level partials; we fake partials
   by interpolating within the last recognized chunk using the WPM prior.
3. **"Invisible to screen recording."** On macOS this relies on private window
   levels; honest answer: replicate via the Obj-C shim with
   `CGShieldingWindowLevel()` — works for most capture APIs, document limits.
   On Linux there is no universal guarantee; offer "capture-invisible" via
   Xcomposite bypass flags best-effort. Market this carefully.
4. **PPTX import.** OOXML zip → `ppt/notesSlides/notesSlideN.xml` → regex-lite
   text extraction. ~300 LoC with miniz. Keynote users export pptx first
   (same workflow as Textream).
5. **Director read-lock.** Server tracks `maxDeliveredWordIdx`; PUT /script
   rejects edits before that index (HTTP 409). Trivial once state is central.

---

## 5. AI-agent layer (our differentiator — use the agent-c-02 design)

The prompter becomes an **agent host**: every show has a copilot talking to
any OpenAI-compatible endpoint (config: `base_url`, `api_key`, `model`):
- **Coach**: consumes the same aligned-word stream; whispers to director UI:
  "pace 210 wpm — slow down", "you said 'um' 7 times in 2 min", "drifted 6s
  behind".
- **Auto-director**: LLM watches transcript progress; when a sponsor segment
  ends it *pushes* the next block through the Director API (read-lock keeps
  delivered text immutable). Human presenters get a live writer.
- **Tele-Q&A**: audience questions arrive via `/ask` webhook; small model
  summarizes them into the prompter queue between pages.
- **Offline field assistant** (Tier-A product from agent-c-01): same binary on
  a Pi with whisper.cpp + local llama.cpp endpoint = prompts, transcribes,
  and answers in the field with zero connectivity.

This is exactly the agent-loop skeleton from `docs/audit/agent-c-02-reference-designs.md`
reused verbatim: tokenize→call→parse→act, except the "action space" is
scroll/lock/push/coach instead of shell tools.

---

## 6. Milestones

- **M0 (weekend)**: core/pacer + SDL fullscreen + classic mode + .textream
  JSON load/save. Usable teleprompter already beats most paid apps.
- **M1 (+1 wk)**: miniaudio capture, VAD voice-activated mode, mouse-scroll
  override, multi-page countdown, mirror flip.
- **M2 (+1–2 wk)**: whisper.cpp integration + align.c word tracking, notch
  overlay, follow-cursor, opacity/glass options.
- **M3 (+1 wk)**: mongoose server, remote viewer HTML+QR, Director Mode with
  read-lock. Phone-as-teleprompter now possible — unique vs Textream.
- **M4 (+1 wk)**: pptx import, update checker, config file, systemd/launchd
  units, CI cross-build matrix.
- **M5 (ongoing)**: AI coach/auto-director via OpenAI-compatible client;
  benchmark table (Pi4/M1/Xeon); WASM demo page.

Effort estimate: ~8–12k LoC total including vendored third_party glue. One
maintainer, weeks not months, because every subsystem above is either
documented in our existing audits (JSON formats, HTTP client, agent loop) or
a known small library.

---

## 7. Positioning vs market (why this wins)

- Free/MIT like Textream, but **runs everywhere** (Linux creators, OBS farms,
  Pi-based studio hardware, church/school AV rigs, Android tablets via web).
- **Director-first**: phone/tablet scripting is native, not a Mac-side add-on.
- **AI copilot** bundled — nobody in the teleprompter market (PromptSmart
  $99+/yr, Teleprompter Premium subscriptions) offers local, private coaching.
- **Hardware angle**: sell a Pi kit (prompter + mic + HDMI mirror glass) as a
  $150 alternative to $2000 professional rigs — open-source software moat.
