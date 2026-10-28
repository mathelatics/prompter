# cPrompter — Project 06: Testing, Build Matrix & Verification Discipline

## 1. The four sanctioned builds (all warning-free under `-Wall -Wextra`)
```sh
make                # default: ZERO deps. core+format+audio(null-capable)+ai+hotkeys
make test           # same sources + tools/test_core.c → ./cp_tests   [93 checks / 0 fail]
make SDL=1          # + ui/overlay_sdl.c        (needs libsdl2; SDL2_ttf autodetected)
make SERVER=1       # + server/http_ws.c server/remote.c vendor/qrcodegen.c (POSIX only)
make WHISPER=1      # + whisper.cpp link from third_party/whisper.cpp/build (cmake-built)
make CURL=1         # -DCP_HAVE_CURL=1 for https cloud endpoints in ai/http_client
```
Makefile facts (verified by reading it): `CFLAGS = -O2 -std=c11 -Wall -Wextra
-Wno-unused-parameter -g`; `LDLIBS = -lm -lpthread -ldl` (+SDL2/+whisper/+ggml as
switched); test binary compiles sources directly with `-DCP_TESTING=1` (exposes statics
where needed); `clean` also sweeps `/tmp/cp_*.textream /tmp/cp_*.wav` fixtures;
`demo` target runs samples/demo.txt at 4 w/s. Feature detection uses pkg-config with
sane fallbacks so a missing ttf doesn't break the build.

**Known Makefile quirk:** object files are not tracked against feature flags — switching
between `make`, `make SDL=1`, `make SERVER=1` without `make clean` can relink stale .o's
("Nothing to be done" while the flag changed). Always `make clean && make <flags>` when
flipping features. (Fix candidate: per-flag OBJDIR suffix.)

## 2. Unit suite (`tools/test_core.c`) — coverage map, 93 checks
- script model: tokenize/pages/headings/est_seconds/page_of/oob NULLs
- pacer: classic advance, pause freeze, voice gating, **track easing semantics**
  (forward-of-pos accepted; behind-pos rejected; tick loop proves monotone ease — these
  encode the *fixed* backtrack-floor behavior, see doc 01 §2), jump/allow_back rules
- read-lock: chars_for_words monotonicity, apply_edit accept-ahead/reject-behind
- formats: .textream save→load round-trip incl. settings; pptx `<a:t>` extractor on
  fixture XML (entities, paragraph breaks); fixture built by make_test_pptx.py
- audio DSP: resample length/rms/hp1 sanity; VAD speech-vs-silence decisions + hangover;
  WPM estimator on synthetic syllable trains
- aligner: exact/norm/fuzzy ladder, short-word exact rule, truncation containment,
  filler drop, skip-ahead beats re-read ("the…the" regression), drift counter, give-up
- track mock backend: true wall-clock pipeline test (scripted words at mock_wps → poll →
  align → pacer moves) — deliberately rate-limited-correct after an honest test bugfix
- hotkey table: every action reachable, wheel/pseudo keys map
- coach rules: fast/slow pace triggers, silence warn, drift→RESYNC, mailbox seq dedupe
- json/openai: find_path/get_string/get_number on real-shaped payloads; cfg_from_env
  defaults + bad-URL rejection (graceful-fallback contract)

## 3. Differential testing (the project's superpower)
QR encoder was validated module-by-module against pip `qrcode` reference matrices
(v1/v2/v8 strings incl. the URL and HELLO WORLD alphanumeric case) — harnesses lived in
/tmp (qrdump.c, rstest.c, qrdiff.py). Lesson recorded in audit doc 03: never trust
eyeballed QR output; read codewords back through your own function-map and diff.
If you touch vendor/qrcodegen.c, RE-RUN that harness before merging.

## 4. Integration tests actually executed (evidence log)
```sh
./cprompter samples/demo.txt --headless --speed 8
  → "3 page(s), 68 words … [classic]" … "[done] delivered 67/68 words, max_delivered=68.0"
timeout 3 ./cprompter samples/demo.txt --speed 8 --coach rules --loop
  → exit 124 (still looping correctly), prints "coach: rules (target 150 wpm)"
make SERVER=1  → links; grep-clean of warnings after final fix pass
```
NOT yet executed (open debt, do first next session): the M3 curl/WS checklist in
doc 03 §5 — requires wiring `--remote/--director` into main.c and adding
tools/ws_probe.py. Everything claimed green above has command output behind it.

## 5. CI plan (not yet created — repo currently has no .github/)
Proposed matrix job per PR: ubuntu-latest × {default, test, SERVER=1} + macos-latest ×
{default, SDL=1}; whisper build cached separately (slow). Commands are exactly §1. Add
`make clean` between flag variants (see quirk §1). Optional ASAN/UBSAN flavor:
`make CFLAGS="-O1 -g -fsanitize=address,undefined …"` then `make test`.

## 6. Reproducing the whole history honestly
Each milestone ended with: build+test outputs pasted into the session, failures either
fixed in code or the assertion rewritten WITH a written justification (audit docs
01–03 record every such decision, including the two moments where wishful tests were
caught and corrected, and the fake-fix heuristic that was removed). When in doubt about
whether something "really works": run doc 03 §5 + §4 commands verbatim.

## 7. Roadmap hooks (from plan §M6+)
Real-microphone field test day (voice+track modes on hardware), macOS Obj-C window-level
shim, WASM demo build, Pi framebuffer port, streaming cloud-ASR backend in track.c,
director UI coach toggle. Each has its seam already marked in docs 02/03/05.
