# cPrompter — Known Issues / Deferred Fixes (M3 carry-over)

Status: **tracked, deliberately deferred** so we can proceed to M4. Nothing here is
hidden; every item was discovered by live testing, not guessed. Do NOT claim M3
"complete" until items 1–5 below are green.

---

## Issue 1 — QR encoder is NOT bit-exact vs ISO/IEC 18004 (BLOCKER for real QRs)

**Symptom.** Differential harness (`/tmp/qrdiff.py` comparing against `pip install qrcode`)
shows the data region scrambled: reading codewords back out of my matrix yields
`40 24 84 3a 46 bb …` while the reference reads `40 24 84 90 ec 11 ec 11 …`.
Format-info bits also mismatch everywhere (consequence of scrambled data + mask).

**What is already verified CORRECT (do not re-touch blindly):**
- Function-pattern *reader* map agrees with reference on v2-M placement order.
- GF(256) Reed–Solomon ECC generation, EC-M tables v1–v10.
- Finder/timing patterns (timing row/col 6 fixed to full length this session).
- Mask penalty rules 1–4 rewritten ISO-exact (rule 3 includes the 4-light-side
  precondition); format bits use mask value alone as the 5-bit data word for level M.
- Version selection across v1–v10 behaves sensibly (25×25 for a URL, 49×49 for 150 chars).

**Root cause (isolated, fix localized).** The zig-zag data traversal in
`place_modules()` skips the wrong modules because the `fn[]` "function area"
reservation marks **too much** as non-data — e.g. it reserves `k<=8` rows/cols
including (8,6)/(6,8) overlap regions differently than ISO's actual data-region walk.
ISO only excludes: finder+separator 8×8 corners, full timing row/col 6, format info
strips, and alignment patterns. Everything else receives data bits.

**Also watch:** an earlier padding bug (terminator → byte-align → EC/11 pad alternation
starting one byte early) was partially addressed; after fixing `fn[]`, re-check that the
terminator is ≤4 zero bits then byte-align, then `EC 11` pads fill only remaining data
codewords (cap = `data_cw*8`).

**Leftover dead code to delete when touching the file:** a stale `VM[]` guess-table and a
dead first-pass `PUSHBIT` block were removed once but re-verify none remain mid-file.

**Acceptance test.** `/tmp/qrdiff.py` PASSes EXACT module-by-module match on ≥3 strings
spanning versions v1 / v2 / v8 (e.g. `http://192.168.1.42:7373`, a 150-char string, a
plain `HELLO WORLD` alphanumeric case). No "bump version" heuristics allowed — that hack
was already caught and removed.

**Files:** `vendor/qrcodegen.c`, `vendor/qrcodegen.h`. Harness: `/tmp/qrdump.c`,
`/tmp/rstest.c`, `/tmp/qrdiff.py`.

---

## Issue 2 — Two compiler warnings in `server/http_ws.c`

1. Misleading-indentation warning (cosmetic; split statements onto separate lines).
2. `usleep` implicit-declaration warning — needs `#define _DEFAULT_SOURCE` before
   includes or switch to `nanosleep`.

Low risk, must be zero-warning before M3 sign-off.

---

## Issue 3 — M3 CLI wiring not done

`main.c` does not yet have `--remote` / `--director` flags calling
`cp_remote_start() / cp_remote_tick() / cp_remote_print_qr() / cp_remote_stop()`.
The Makefile `SERVER=1` hook already points at the right source files
(`server/http_ws.c server/remote.c vendor/qrcodegen.c`).

---

## Issue 4 — M3 live integration tests never executed

Required before claiming M3 done:
- Start binary headless with `--remote --director`.
- `curl` checks: `GET :7373/` → 200 HTML viewer; `GET :7373/api/state` → valid JSON;
  `GET :7575/` → director page; `POST :7575/api/script` editing a *delivered* word →
  **HTTP 409**; same POST from `:7373` → **403**.
- Small WS client (C or python) counting ~10 Hz pushes from `cp_remote_tick`.
- Full gate: `make clean && make test && make && make SERVER=1` all green.

Note: `server/remote.c` (~360 LoC) compiles and links into the SERVER=1 binary but has
never been exercised over a real socket. Treat as unverified.

---

## Issue 5 — Whisper.cpp backend not vendored/built

`make WHISPER=1` expects `third_party/whisper.cpp` built via cmake. Not present in this
environment, so M2 word tracking runs only through the mock pipeline + align unit tests
(84 checks green). Real-model end-to-end tracking is untested. Acceptable for now because
align.c logic is validated against synthetic ASR streams; revisit before public release.

---

## Minor / polish backlog (non-blocking)

- SDL overlay: mirror-flip (H/V/180°) and fullscreen-Sidecar paths exist but untested
  on real displays (no display in CI). Planned to be covered in M4.
- `.textream` JSON: no schema-version migration path if format evolves.
- Director UI: read-lock indicator shows count but not clickable jump-to-locked-boundary.
- Security docs: advise changing default ports 7373/7575 and closing servers post-show
  (README task, M4).

---

**Rule reminder (from earlier audit):** no dummy/stub code in place of tests; every
completion claim must be backed by an executed command output. This file exists precisely
so deferred work stays visible instead of silently "done".
