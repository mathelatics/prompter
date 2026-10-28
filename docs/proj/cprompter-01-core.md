# cPrompter — Project 01: Core (M0) — script model, pacer, engine

Files: `core/script.[ch]` (241 L), `core/pacer.[ch]` (358 L), `core/engine.[ch]` (349 L).
This is the only layer every other module depends on. Read top→bottom once; then you can
modify anything else without re-reading it.

## 1. Data model — `core/script.h`
**Invariant: Script = Pages = Words.** One flat `cp_word` array with pages storing
`[w0, w1)` ranges into it → O(1) cursor math everywhere ("what page is word N" is a
binary search over ≤ dozens of pages; "where does page P start" is direct indexing).

```c
typedef struct cp_word { char *text; size_t n; int para; float wpm_prior; } cp_word;
typedef struct cp_page { char name[64]; int w0, w1; cp_mode mode;
                         double scroll_speed; int countdown_sec; } cp_page;
struct cp_script { words/capw, pages/capp, char *raw /* full original text */,
    default_mode, default_speed, lang[16], mirror, font_family, font_size,
    highlight_color };
```
- `CP_MODE_CLASSIC|VOICE|TRACK` enum lives here; per-page `mode` overrides default.
- `raw` is kept verbatim for save fidelity and for the read-lock char mapping (§3).
- `para` = paragraph index within page → renderer line-wraps per para.
- `wpm_prior` reserved for per-word duration weighting (currently 1.0 everywhere).

`cp_script_load_text(s, text, split_pages)`: tokenizes whitespace-delimited; `\n\n` or
lines starting `# ` start a new paragraph; if `split_pages`, `# ` headings also start a
NEW PAGE (name = heading text). This is why `samples/demo.txt` loads as 3 pages.
Queries: `cp_word_at` (NULL oob), `cp_page_of`, `cp_script_est_seconds(s, wps)`.

## 2. Pacer — `core/pacer.c` (THE state machine; single source of truth)
State: `pos` (fractional word index), `max_delivered` (monotone high-water mark),
`paused`, `speed_wps`, `t_last`, plus voice fields (`vad_active`, `measured_wpm`) and
track fields (`track_target`, `drifted_by`). Everything downstream — SDL renderer,
10 Hz remote sync, director read-lock, AI coach — consumes ONLY these numbers.

### `cp_pacer_tick(p, now_sec, dt)` per-mode semantics
- **classic**: `pos += speed_wps*dt` unless paused.
- **voice**: advances only while `vad_active`; rate = `max(min_speed, measured_wpm/60)`
  so silence freezes the scroll (Textream behavior). Fed via `cp_pacer_feed_vad()`.
- **track**: eases `pos` toward `track_target` (ASR-aligned cursor) at up to
  ~3 words/frame; fed via `cp_pacer_feed_align()`. **Semantic fix (honest, from M2):**
  backtrack floor is bounded by BOTH `max_delivered+1` AND current display `pos` —
  prevents reversed scroll when the aligner reports a re-read behind where we already
  are (see test: forward-of-pos accepted, behind-pos rejected, tick loop proves easing).
- `dt<=0` no-op; pause just refreshes `t_last`.

### API (all in pacer.h; exact signatures verified)
`init/set_speed/tick/scroll_by/jump(idx, allow_back)/set_paused/feed_vad(active,inst_wpm)
/feed_align(pos)/resync(word_index clamp)/word_index(floor clamped)/finished/
remaining_seconds`. `cp_pacer_set_speed` clamps to Textream range 0.5–8 wps, invalid
values ignored (added during M5 wiring; engine's mutex-guarded setter calls through it).

## 3. Director read-lock — `cp_script_chars_for_words()` + `cp_pacer_apply_edit()`
The lock answers: *"which prefix of `raw` have the audience already heard?"* It maps a
delivered-word count to a char offset by walking the tokenizer's own segmentation
(0 before speech starts, `strlen(raw)` when done). `apply_edit(new_text)` re-tokenizes
in place; returns **-1 + err message if the edit touches chars < max_delivered** —
server turns that into HTTP 409. This is unit-tested directly (no server needed).

## 4. Engine — `core/engine.c` (thread-safe runtime façade)
Opaque `cp_engine` bundles: pacer ptr + script ptr + optional `cp_audio` capture thread
+ `cp_vad` + `cp_wpm` + `cp_track` + `cp_align`, all under one `pthread_mutex`.
- Params struct: `{dev, enable_audio, sensitivity, track(cp_track_params)}`;
  `"null"` device = silent CI backend. Construction failure fills `err` (main.c falls
  back classic-clock on error rather than dying).
- Capture callback (audio thread): VAD push → WPM push → `feed_vad`; whisper/mock poll
  happens UI-side via `cp_engine_update_track()` (once per frame) → aligner → feed_align.
- Getters safe from any thread: `pacer(), script(), speaking(), level(), wpm(), drift(),
  last_heard()`. Controls: `set_speed/toggle_pause/scroll/jump/cycle_mode/
  vad_recalibrate/sensitivity/resync`. Clock: `cp_engine_now()` = CLOCK_MONOTONIC secs.
- **M3 plumbing block** (`engine.c` §"remote/director"): `cp_engine_snapshot(e, out)`
  fills `cp_engine_state{pos,max_delivered,speed_wps,nw,word_index,page_index,paused,
  speaking,finished,level,wpm,mode}` in ONE lock acquisition (atomic view for the 10 Hz
  push); `cp_engine_apply_edit` forwards to pacer; `cp_engine_after_edit` re-points the
  aligner/tracker at the mutated script (cursor preserved by word index).

## 5. main.c headless loop (the reference consumer)
Per iteration: monotonic clock → `cp_pacer_tick` → (coach, see doc 05) → on word change
print ANSI window `[prev3 · HIGHLIGHTED-current · next7]` with dimmed delivered text →
`usleep(8ms)` (~125 fps ceiling) → on finish break or re-`init` if `--loop`.
Exit prints `delivered N/M words, max_delivered=X` (asserted by integration tests).
SDL path (when built & tty & !headless): builds `cp_engine_params` from mode
(voice→mic audio, track→whisper backend), graceful fallback to classic clock on engine
failure, hands control to `cp_overlay_sdl_run(eng, loop, style)`.

## 6. Gotchas for future devs
- Never mutate `cp_script` words/pages outside load/edit paths — pacer holds raw ptrs.
- `pos` is fractional; `word_index` is its floor clamped to nw. Use `max_delivered`
  (not `pos`) for anything irreversible (read-lock, analytics).
- The engine mutex covers audio-thread writes; UI getters copy scalars out — do not
  return pointers into pacer internals from new getters.
- `cp_engine_new` expects the caller-owned `cp_pacer` (stack OK) — engine borrows it.
