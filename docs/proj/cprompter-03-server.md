# cPrompter — Project 03: Server (M3) — Remote Viewer :7373 + Director :7575

Files: `server/http_ws.[ch]` (421 L), `server/remote.[ch]` (395 L),
`vendor/qrcodegen.[ch]` (396 L). Built with `make SERVER=1`. Pure POSIX sockets —
civetweb was evaluated and deliberately NOT needed at this scale (localhost/LAN, few
clients, 10 Hz pushes).

## ⚠️ Honest status note (verified 2026-10-28 during docs pass)
The M3 *engine-side* plumbing (`cp_engine_snapshot`, `apply_edit` read-lock,
`after_edit`) is complete and unit-tested inside the 93-check suite. The server modules
compile and link into the SERVER=1 binary. **However:** CLI flags `--remote/--director`
were never wired into `main.c` (grep proves no `cp_remote_start` call site), so the
live curl/WebSocket integration tests promised in the plan have not been executed.
Treat "runtime behavior" below as *designed-and-compiled*, pending that wiring task
(tracked here and in docs/audit/cprompter-03-known-issues.md style). Do not claim M3
field-tested until someone runs the §5 checklist for real.

## 1. HTTP+WS core — `server/http_ws.c`
Thread-per-connection accept loop over one listening socket. Own SHA-1 + base64
(verified constants) implement the RFC6455 handshake (`Sec-WebSocket-Accept`). Handles:
GET/POST with NUL-terminated body buffering, query-string split, Content-Type capture,
CORS headers, text frames (FIN+opcode 0x1), ping→pong, close frames, client tracking.
Handler API (`cp_http_req`): fill `status` (0→200), `ctype/out/out_cap/out_len`;
`want_ws=1` when the client requested an upgrade → connection joins broadcast group.
Broadcast: `cp_httpd_ws_broadcast(d, text)` + `cp_httpd_ws_count(d)`; optional
`cp_httpd_on_ws_open` callback. No TLS by design (LAN-only feature; document says so).

## 2. Routes — `server/remote.c` (two cp_httpd instances around one cp_engine)
**Viewer :7373 (read-only):**
- `GET /` → embedded single-page HTML teleprompter view (word highlight, progress bar,
  wpm/pause/mic status; polls or WS-subscribes).
- `GET /api/state` → JSON snapshot (polling fallback for non-WS clients).
- `GET /qr.svg` → QR of the viewer URL rendered as SVG via vendor encoder.
- `POST *` → **403** (viewer must never control the show).
- `WS /ws` → 10 Hz state pushes.

**Director :7575 (write + control):**
- `GET /` → editor page: textarea, Push/Pause/Speed/Jump buttons, live state line,
  read-lock indicator `"read-lock: first N words can't be edited"` (N = max_delivered).
- `POST /api/script {text}` → `cp_engine_apply_edit`; on conflict replies
  **409 `{"error":"read-lock: <msg>"}`** (message JSON-escaped via jesc); on success
  `cp_engine_after_edit` + broadcasts new state to BOTH groups (script_changed shape).
- `POST /api/pause|speed|jump` → engine controls, then same dual broadcast.
- `WS /ws` → state stream incl. delivered-text info.

JSON builder `state_json(eng, include_script, buf, cap)` serializes the atomic
`cp_engine_state` snapshot (+ script text for director payloads). Buffers are 64 KiB on
stack — fine for scripts under ~60k chars; revisit if huge scripts become a use case.

## 3. 10 Hz tick
`cp_remote_tick(r)` called from the UI thread every frame; internally throttled with
CLOCK_MONOTONIC to ≤1 push per 100 ms (Textream parity: word highlighting, waveform
level, progress at ~100 ms latency). Also used to update viewer/director counts.

## 4. QR codes — `vendor/qrcodegen.c`
Genuine ISO 18004 byte-mode encoder, EC level M, versions 1–10: GF(256) Reed–Solomon,
finder/timing(full row&col 6)/alignment patterns, format strips, correct zig-zag data
walk (fn[] reservation fixed to ISO-exact set), all 8 masks scored by the 4 penalty
rules (rule 3 includes the 4-light-side precondition; rule 4 step math), BCH format
info. **Differential-tested bit-exact vs pip `qrcode` reference matrices on ≥3 strings
spanning v1/v2/v8** (this closed the last known QR bug — see audit doc 03).
API: `cp_qr_encode(text) → cp_qr{size,modules}`, `cp_qr_print` (terminal, half-blocks),
`cp_qr_free`. `cp_remote_print_qr` resolves host IP (env `CP_HOST` override →
gethostname+gethostbyname+inet_ntoa) and prints both URLs as scannable ASCII QRs;
phones scan straight off the desktop.

## 5. Required live-test checklist (the honest definition of M3-done)
```sh
make SERVER=1 && ./cprompter samples/demo.txt --headless --loop &   # after wiring flags
curl -s localhost:7373/            | grep -i '<title>'              # expect viewer HTML 200
curl -s localhost:7373/api/state                                    # expect JSON fields
curl -s -X POST localhost:7373/api/pause                            # expect 403
curl -s -X POST localhost:7575/api/speed -d 'wps=5'                 # expect 200 JSON
curl -s -X POST localhost:7575/api/script -d 'text=<early words>'   # expect 409 read-lock
python3 tools/ws_probe.py ws://localhost:7373/ws                     # count ~10 msgs/sec
```
(ws_probe.py does not exist yet — write it as part of finishing this task.)

## 6. Design decisions log
- Hand-rolled server over civetweb: −1 dependency, full control of WS framing; OK
  because threat model is a private LAN stage, not the internet. Never expose these
  ports through a firewall/NAT without a reverse proxy doing auth.
- Ports match Textream defaults (7373/7575) intentionally: muscle-memory parity.
- Two separate httpd instances instead of vhosts: simpler isolation of read/write
  semantics; the 403-on-viewer-POST rule falls out structurally.
