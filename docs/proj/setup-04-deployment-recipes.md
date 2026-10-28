# Setup 04 — Production / Studio Deployment Recipes

For running cPrompter as real gear: stage shows, streaming rigs, Pi teleprompter
hardware, and shared studio setups. Commands assume the builds from setup-01.

## Recipe A — Raspberry Pi hardware teleprompter (the $150 rig)

Target: Pi 4/5 with a 7–10" HDMI panel + beam-splitter glass, phone-as-director.

```sh
# on the Pi (64-bit OS): build natively, zero deps needed for classic mode
git clone <your-repo> && cd <repo>/cprompter
make                                   # core binary, ~350 KB
sudo apt install libsdl2-dev && make clean && make SDL=1     # overlay
mkdir -p models && curl -L -o models/ggml-tiny.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
make clean && make WHISPER=1 SERVER=1  # full rig build
```

- Launch on framebuffer/console without a desktop: run inside `weston --scale=1`
  kiosk session or plain X with compositor disabled; fullscreen mode + mirror
  flip (horizontal) turns the panel into a prompter glass feed (hotkey table in
  docs/proj/cprompter-02-audio-ui.md).
- USB condenser mic directly on the Pi drives voice/track modes.
- tiny.en tracks acceptably on Pi 5 (4 cores); on Pi 4 prefer voice mode for
  long sessions (VAD costs ~single-digit % CPU vs ASR load).
- Autostart via systemd unit:

```ini
# /etc/systemd/system/cprompter.service
[Unit]
Description=cPrompter studio rig
After=graphical.target sound.target

[Service]
User=cp
WorkingDirectory=/home/cp/rig          # so ./models/*.bin resolves
ExecStart=/home/cp/rig/cprompter /home/cp/rig/show.textream --mode track --loop --coach rules
Restart=on-failure

[Install]
WantedBy=graphical.target
```

## Recipe B — OBS/streaming laptop (Linux or macOS)

Goal: prompter under the webcam line, invisible to capture.

```sh
make clean && make SDL=1
CCP_BASE_URL=http://127.0.0.1:8080/v1 CCP_MODEL=qwen2.5-7b-instruct \
  ./cprompter sponsor-read.textream --mode track --coach llm
```

- Overlay window sits near camera housing; verify invisibility once per rig by
  recording 10 s in OBS and scrubbing (window-level behavior documented in
  docs/proj/cprompter-02-audio-ui.md; Wayland caveat in setup-03).
- Run a local llama.cpp server (`--api --port 8080`) so the LLM coach adds no
  cloud dependency mid-stream; advice lands on stderr/overlay, never blocks the
  pacer thread.
- Keep a classic-mode fallback tab open: if ALSA/CoreAudio dies mid-show, hotkey
  **m** cycles back to clock-driven scroll instantly.

## Recipe C — Two-person director workflow (phone + laptop)

Once the `--remote/--director` CLI wiring lands (open item tracked in
docs/proj/cprompter-03-server.md §5 — until then this recipe runs against any
build containing that wiring):

1. Laptop: `./cprompter show.textream --remote --director` → QR prints in
   terminal; both servers bind LAN (:7373 viewer, :7575 director).
2. Talent's tablet/phone scans QR → read-only word-tracked view at 10 Hz.
3. Director opens :7575, writes/pushes segments live; delivered text is
   read-locked (HTTP 409) so the talent can never be gaslit by retro-edits.
4. Firewall: allow inbound TCP 7373/7575 only on the trusted interface
   (`sudo ufw allow in on lo,eth0 to any port 7373,7575 proto tcp` equivalent);
   shut servers after the show. Ports are function args to `cp_remote_start()`
   if you must move off Textream-compatible defaults.

## Recipe D — Conference keynote (no network at all)

Air-gapped requirement stack:

```sh
make clean && make SDL=1 WHISPER=1      # NO CURL=1, no cloud anything
./cprompter keynote.pptx --mode track --coach rules --target-wpm 140
```

- pptx import pulls presenter notes straight off the USB stick.
- Rules coach gives pace coaching offline; nothing queries any socket except
  localhost. Verified privacy stance: see setup-03 FAQ "Does anything leave my
  machine?"
- Rehearse first pass at `--speed 3.5`; save position-bearing `.textream` via
  hotkey **s** so day-of resumes your last page.

## Recipe E — Podcast producer desk

- Voice mode + `--loop` for ad-read drills; producer monitors progress over the
  remote viewer (Recipe C) and pushes new reads through Director Mode during
  natural pauses.
- Multi-page scripts auto-surface next-topic countdown via page boundaries
  (`[ ]` hotkeys jump pages manually).
- Long-session tuning (from plan §adv): Classic mode for 2 h+ marathons keeps
  CPU cool; drop overlay opacity when compositing on weak GPUs.

## Update & rollback policy for venues

1. Ship tagged release binaries (one per flavor), not source checkouts, to
   venue machines; keep `make test` green gate in CI before tagging
   (docs/proj/cprompter-06-testing-builds.md).
2. Scripts (.textream) are forward/backward compatible JSON — archive them per
   show; they carry settings, so a newer binary reading an older file is safe.
3. Rollback = swap binary, same script dir. No migrations exist by design.

## Known deployment gaps (be honest with users)

- Windows .exe: not built yet (WSL2 only today) — don't promise native binaries
  until CI matrix proves it.
- `--remote/--director` CLI flags: pending main.c wiring (see Recipe C note).
- Notarized macOS app bundle: out of scope for v1; document Terminal launch.
