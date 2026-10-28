# cPrompter — Project 04: Formats (M0 .textream + M4 pptx) & Tools

Files: `format/script_io.[ch]` (~330 L), `format/pptx.[ch]` (~250 L),
`tools/test_core.c`, `tools/make_test_pptx.py`, `samples/demo.txt`.

## 1. `.textream` save format — `format/script_io.c`
JSON shape chosen to mirror the Textream blog's described format (portable scripts):
```json
{ "version":"1.0",
  "pages":[ { "id":1, "content":"...", 
              "settings":{ "guidanceMode":"wordTracking"|"classic"|"voiceActivated",
                           "scrollSpeed":3.0, "speechLanguage":"en-US" } } ],
  "metadata":{ "created":"...", "lastModified":"..." },
  "settings":{ "mirror":0, "fontFamily":"sans", "fontSize":24, "highlightColor":"yellow" } }
```
- Hand-rolled minimal reader/writer (jsmn-style scanner in ~this file only) — keeps the
  default build dependency-free; parson is the designated swap-in if it ever grows.
- Writer emits per-page content by slicing `raw` at word boundaries → reload fidelity
  round-trip is unit-tested (save→load→same pages/words/settings).
- Reader failure path in main.c: parse error message printed, then **falls back to
  treating the file as raw text** (never refuses to run a show).
- guidanceMode strings map to cp_mode enum; unknown values → classic (forward-compat).
- `cp_script_save_txt` convenience exporter too. Hotkey `s` saves mid-show.

## 2. PowerPoint import — `format/pptx.c` (M4)
A `.pptx` is a ZIP of OOXML; presenter notes live in `ppt/notesSlides/notesSlideN.xml`.
Implementation choices (verified against generated fixtures):
- Uses **system `unzip -p` via popen** (documented requirement; no vendored zip lib —
  miniz remains the upgrade path if we need Windows-native without unzip binary).
- Extractor `cp_pptx_extract_at_texts(xml, len)` (exposed for unit tests): concatenates
  all `<a:t>` runs, inserts `\n` at `</a:p>` paragraph boundaries, decodes entities
  (&amp; &lt; &gt; &quot; &apos; and numeric &#xNN;). malloc'd return.
- One slide's notes = one page (page name = "Slide N"). Ordering by slide number parsed
  from filenames, not zip order (zip order is not guaranteed!).
- CLI: `./cprompter deck.pptx` auto-detects by extension; errors like missing unzip or
  no notes surface as `pptx import: <msg>` exit 1. Keynote/Google Slides users export
  to .pptx first (Textream parity instruction).
- Fixture generator `tools/make_test_pptx.py` builds a minimal valid pptx (hand-written
  XML parts + zipfile module) so CI never needs PowerPoint itself.

## 3. samples/demo.txt
3-page markdown-ish script (`# ` headings split pages) used by README examples,
`make demo`, headless integration checks, and coach demos. ~68 words — short enough for
fast CI runs at `--speed 8`.

## 4. tools/test_core.c
Single-file unit harness (no framework by design): tiny CHECK macros, groups covering
script model, pacer modes incl. track easing semantics, read-lock char mapping, apply_edit
accept/reject cases, .textream round-trip, pptx extractor, VAD/WPM on synthetic frames,
aligner edge cases ("the…the" skip-ahead vs backtrack, short-word exact rule, truncation
containment), mock track pipeline (true wall-clock loop), hotkey table, coach rules engine,
json/openai payload parsing. Current count: **93 checks, 0 failures** (`make test`).
When adding behavior: add the check HERE first; tests encode actual verified behavior.

## 5. M4 UI polish shipped alongside (in ui/overlay_sdl.c, see doc 02 §7)
- Mirror flip f: none/horizontal/vertical/180° cycle persisted into style settings
  (`s->mirror`, saved through .textream `settings.mirror`).
- Fullscreen + notch styles with 'o' toggle; fixed-display/follow-mouse behaviors noted
  as designed in plan; real macOS window-level pinning still needs the Obj-C shim.
- Style persistence bug found & fixed during M4 wrap-up: `style_load` was not restoring
  rotation field (fixed; regression covered by save/load round-trip check).

## 6. What is NOT done (honest list, carried forward)
- `--remote/--director` CLI wiring + live curl/WS tests (doc 03 §5 checklist).
- GitHub Actions cross-build matrix (planned in M4 originally; repo has no .github yet).
- WASM/Emscripten browser build (plan Tier-1 funnel idea; untouched).
- Follow-cursor warp polling (needs platform-specific mouse queries; SDL cursor pos OK).
