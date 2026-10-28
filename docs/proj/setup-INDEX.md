# Setup & Run Documentation — Index (docs/proj/setup-*.md)

New second set of project docs: **operational** guides (how to install, build,
run, debug, deploy), complementing the seven existing architecture/codebase
docs (`cprompter-00..06`). Written 2026-10-28; every command and expected output
in these files was executed or grep-verified against `/workspace/cprompter` at
writing time.

## Files & when to read them

| File | Audience / moment | Contents |
|---|---|---|
| `setup-00-prerequisites.md` | before touching a machine | OS/hardware baseline, mandatory-vs-optional deps table (default build = zero external libs), complete env-var list with source line refs (`CCP_BASE_URL/CCP_MODEL/CCP_API_KEY/OPENAI_API_KEY/CP_NULL_TONE`), disk layout (`models/ggml-base.en.bin` CWD-relative!), ports 7373/7575, known setup-affecting debt |
| `setup-01-build.md` | first compile + every flag change | all four build flavors with verified commands & outputs (`make test` → 93 checks 0 failures; SERVER=1 link proof via `nm`), whisper.cpp third_party setup, SDL_ttf autodetect, CURL option, stale-.o gotcha #1 |
| `setup-02-run-first-session.md` | "it built, now what" | classic terminal run, `.textream` save/load round-trip (verified 3 pages/68 words), voice mode + full hotkey list, track mode incl. model download URLs, coach rules vs llm with local-server and cloud examples, planned remote UX, macOS/Windows notes |
| `setup-03-troubleshooting.md` | anything misbehaves | symptom→cause→fix tables keyed to real code strings (e.g. `audio/audio.c:111 libasound not found`, `main.c:178 bad CCP_BASE_URL`), VAD calibration steps, CP_NULL_TONE diagnostic trick, Wayland overlay caveat, data-format fixes, privacy/embedding FAQ |
| `setup-04-deployment-recipes.md` | taking it to a venue | Pi teleprompter-hardware rig (+systemd unit), OBS streaming laptop w/ local LLM coach, two-person director workflow, air-gapped keynote stack, podcast producer desk, release/rollback policy, honest gaps list (no Windows .exe yet, --remote CLI wiring pending) |

## Reading order for newcomers

1. setup-00 (check your box) → setup-01 (`make test` then `make`)
2. setup-02 §1–2 (first scroll + file format) — 5 minutes to something useful
3. setup-02 §3–5 as you adopt voice/track/coach
4. keep setup-03 bookmarked; setup-04 when shipping to real shows

## Cross-references into the architecture set

- Module internals behind each run behavior: `cprompter-01-core.md` (pacer/engine),
  `cprompter-02-audio-ui.md` (VAD/WPM/align/overlay/hotkeys),
  `cprompter-05-ai-coach.md` (mailbox, ACTION protocol, env config).
- Server route semantics & open wiring checklist: `cprompter-03-server.md` §5
  (referenced honestly from setup-00/02/03/04 so nobody expects unshipped flags).
- Test-suite map & CI matrix: `cprompter-06-testing-builds.md`.
- Historical bug log / deferred issues: `docs/audit/cprompter-03-known-issues.md`.

## Maintenance rules (same discipline as the code)

- Any new CLI flag, env var, port, or default path MUST be added to setup-00's
  tables in the same PR that ships it — docs drift is treated as a bug.
- When `--remote/--director` wiring lands, update setup-02 §6, setup-03, and
  setup-04 Recipe C to remove their ⚠️ caveats, and tick cprompter-03-server §5.
- Re-verify quoted outputs ("93 checks", sizes, sample counts) whenever tests or
  samples change; never copy numbers forward unchecked.
