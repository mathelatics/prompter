# cPrompter — Project 05: AI Coach / Auto-Director (M5)

Files: `ai/json.[ch]` (~210 L), `ai/http_client.[ch]` (~290 L), `ai/openai.[ch]`
(~135 L), `ai/coach.[ch]` (~295 L). CLI: `--coach rules|llm`, `--target-wpm <n>`;
env: `CCP_BASE_URL` / `CCP_MODEL` / `CCP_API_KEY`. This is the differentiator vs Textream:
a private, local-first presentation copilot that works with ANY OpenAI-compatible server.

## Architecture — two layers, one mailbox (`ai/coach.c`)
```
UI frame ── cp_coach_update(s, pacer, wpm_now, speaking, now)
   │
   ├─ RULES ENGINE (synchronous, zero-latency, always on when --coach given)
   │     pace too fast/slow vs target_wpm → SLOW_DOWN/SPEED_UP
   │     aligner drift streak > drift_threshold(5) → RESYNC hint
   │     silent-but-unpaused > silence_warn_sec(8) → TEXT nudge
   │     page nearing end → (context for auto-director)
   │
   └─ LLM ADVISOR (only if enable_llm; throttled ≥ llm_every_sec default 45 s)
         snapshot state + recent heard-words window → worker thread →
         POST /chat/completions → parse "ACTION:<verb>" line → advice kind
post(advice_kind, text, at) → single-slot MAILBOX {kind,text[192],at,seq}
UI frame ── cp_coach_poll(&a,&last_seq)  (mutex-guarded, dedupes by seq)
```
The coach is **read-only toward the pacer**; actions are applied by the HOST exactly as
a human director would (main.c headless loop verified behavior):
`SLOW_DOWN→speed*0.85 · SPEED_UP→speed*1.15 · PUSH_NEXT→jump(next page,w0,allow_back) ·
HOLD→pause · RESYNC/TEXT→banner only`. Same API surface remote/director uses — no
privileged path exists for the model. Worker thread never blocks the render loop; a dead
endpoint just yields no LLM advice (rules keep working). Bad `CCP_BASE_URL` at startup →
stderr warning + automatic fallback to rules-only (tested graceful-degradation rule).

## Advice protocol with the model (system prompt in coach.c)
Compact state summary (page, delivered/total words, wpm, drift, mode) + last ~40 heard
words; reply format: one ≤160-byte coaching sentence, optionally ending in
`ACTION:slow_down|speed_up|resync|push_next|hold`. Parser = substring scan of
`ACTION:` verbs (deliberately dumb & robust to chatty models; unknown → plain TEXT).
Temperature low; timeout from cfg (default 15 s per request).

## `ai/http_client.c` — minimal blocking HTTP/1.1 client over POSIX sockets
Purpose-built for chat-completions: `cp_http_post_json(url, body, bearer, timeout_ms,
resp)` and `cp_http_get(...)`; resp={status, malloc'd body, err[256]}. No keep-alive,
**no hand-rolled TLS**: `https://` URLs delegate to optional curl (`make CURL=1` →
`CP_HAVE_CURL=1`; without it https fails with clear error while http:// — covering
llama.cpp/Ollama/our engine on localhost/LAN — works natively). Whole-exchange deadline
via SO_RCVTIMEO + connect timeout. Chunked-transfer tolerated; Content-Length primary.

## `ai/openai.c` — OpenAI-compatible glue (mirrors docs/audit/agent-c-02 design)
`cp_openai_cfg{base_url[256],model[128],api_key[256],timeout_ms}`;
`cp_openai_cfg_from_env` defaults base to `http://127.0.0.1:8080/v1`, empty key → no
Authorization header (correct for local servers). Empty model → probes `GET /models`
and takes first entry. `cp_openai_chat(cfg, system, user, temp, out, outn, err)` builds
the messages JSON (string-escaping done carefully — quotes/newlines/backslashes), posts
`/chat/completions`, extracts content via json path.

## `ai/json.c` — tiny extractor, not a validator
`cp_json_get_string(json,key,out)` first-match + unescape (\uXXXX basic);
`cp_json_find_path("choices.0.message.content")` dotted objects/array indices;
`cp_json_get_number(path, def)`. ~200 LoC, correctness exercised against real-shaped
OpenAI/Ollama payloads in the unit suite. Swap-in point: parson if we ever parse big trees.

## UI surfacing
- Headless loop: cyan `[coach] <text>` line on stderr, ANSI-clear-safe with the scroll.
- SDL overlay: banner strip under the word window (doc 02 §7), auto-fades after a few
  seconds (age checked against advice.at).
- Test hook: `heard_words` NULL-terminated array in cp_coach_params lets tests inject an
  ASR transcript without audio; `cp_coach_llm_once()` runs one synchronous round-trip
  for deterministic assertions against a stub server.

## Known limitations / next work (honest)
- Live LLM integration test used a local stub server; a run against a real llama.cpp
  endpoint should be logged before shipping the flag outside dev machines.
- Advisor speaks English only today (prompt + ACTION verbs); i18n = prompt templates.
- No streaming/token-by-token coach replies (unnecessary: one short line per 45 s).
- Consider letting the director web UI toggle coach on/off mid-show (needs one more
  engine control + route).
