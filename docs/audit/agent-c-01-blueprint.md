# Agent-C Blueprint — AI Agent Systems Purely in C (OpenAI-Compatible API + Local llm.c)

> Goal: build market-ready agentic products where the **entire agent runtime is plain C**.
> Model calls go over HTTP to any OpenAI-compatible endpoint (`base_url` + optional `token`):
> - **Local**: llama.cpp server, ollama, vLLM, TGI, or our own llm.c inference engine → `http://127.0.0.1:8080/v1`
> - **Cloud**: OpenAI / Anthropic-compat / Groq / Together / Fireworks / OpenRouter → `https://…/v1` + `Authorization: Bearer <key>`
> Our llm.c/gpt.c audit docs give us the *local* side (tokenizer, sampler, forward pass, KV-cache inference ideas) so we can even self-host the "brain" with zero dependencies.

---

## 1. Why C for agents? (market pitch)

| Pain point today | What a pure-C agent stack fixes |
|---|---|
| Python agent frameworks (LangChain/AutoGen/CrewAI) = 300+ transitive deps, slow cold start, security surface | Single static binary, ~1–5 MB RAM floor, boots in <10 ms, tiny CVE surface |
| Agents on edge/IoT/routers/NAS are impossible with Python runtimes | Cross-compiles to ARM/RISC-V/MIPS; runs where no interpreter exists |
| Cloud agent hosting bills per-second of a fat container | A C process fits in 64 MB Lambda-equivalent; 50–100× cheaper fleet |
| Auditing/compliance can't trace what a framework did | Deterministic loop, everything logged by design (see §4) |
| Latency: python serialization + framework overhead adds 50–300 ms/call | Raw libcurl + hand-parsed JSON ≈ single-digit ms overhead |

Target: **"Deno-for-agents" moment** — one binary, one config file, any OpenAI-compatible brain.

---

## 2. Core architecture (the reusable kernel of every project below)

```
        ┌──────────────────────────────────────────────────────┐
        │                    agentd (C binary)                 │
        │                                                      │
 config │  ┌────────┐   ┌───────────┐   ┌───────────────────┐  │   HTTPS    local:8080/v1
 (toml)─┼─▶│ Loop   │──▶│ Prompt/   │──▶│ LLM Client        │──┼──▶ OpenAI-compatible API
        │  │ Engine │◀──│ ChatML fmt│◀──│ (libcurl+cJSON/    │◀─┼──  (cloud key OR base_url)
        │  └───┬────┘   └───────────┘   │  jsmn streaming)  │  │      chat/completions SSE
        │      │                        └───────────────────┘  │
        │      ▼                                                │
        │  ┌──────────┐  ┌──────────────┐  ┌────────────────┐  │
        │  │ Tool     │  │ Memory       │  │ Audit log      │  │
        │  │ Registry │  │ (sqlite/     │  │ (append-only   │  │
        │  │ (plugins │  │  lmdb/vector)│  │  hash-chain)   │  │
        │  └──────────┘  └──────────────┘  └────────────────┘  │
        └──────────────────────────────────────────────────────┘
```

### 2.1 Minimal module list (each ≤ ~500 LoC of C)
1. **`llm_client.c`** — libcurl easy-handle wrapper. Config: `base_url`, `api_key`, `model`, `timeout`. Implements `POST /v1/chat/completions` (blocking + SSE streaming via write-callback), parses `tool_calls[]` with jsmn (no full JSON DOM needed). This is the *only* network code; everything else is transport-agnostic.
2. **`loop.c`** — the ReAct/function-calling state machine: `while(!final_answer && steps<N){ call_llm(); parse tool_calls; execute; append observations; }`. Same skeleton you'd get from our llm.c sampler loop — tokenize→forward→sample becomes prompt→request→parse.
3. **`tools.c`** — registry: `{name, json_schema, fn}`. Built-ins: `shell` (fork/exec with rlimit+seccomp), `http_get`, `file_read/write` (sandboxed root), `python` (optional escape hatch). Plugins = dlopen'd `.so` exporting `agent_tool_*`.
4. **`memory.c`** — session store: sqlite3 (chat history), plus optional vector recall using embedding calls (`/v1/embeddings`) and brute-force cosine in C (fine for <100k vectors; small models like nomic-embed / Granite embeddings are cheap).
5. **`guard.c`** — input/output filters: regex + keyword blocklists, length caps, tool-allowlist per profile, max-cost budget per session. Optionally call a *second small model* as judge (Llama Guard 3 1B / Granite Guardian via base_url).
6. **`audit.c`** — append-only JSONL log with SHA-256 hash chain (tamper-evident transcript for compliance).
7. **`config.c`** — TOML-ish parser (or ini.h, single header) defining profiles: which model/base_url/token/tools/budget each agent gets.

### 2.2 Local-brain option (where llm.c code pays off directly)
- Run our **inference engine** (Tier-2 idea #8 from previous brainstorm: training forward pass + KV cache) behind a 500-line C HTTP server exposing `/v1/chat/completions` → the agent binary literally cannot tell local vs cloud apart.
- Use llm.c's tokenizer + xorshift/mt19937 sampler docs (`llm-c-04`) for deterministic sampling — reproducible agent traces, gold for testing/CI.
- Tiny specialized brains: GPT-2-class distilled routers (classify intent → pick big model) trained with train_gpt2 recipe = pennies, runs on CPU reference trainer (`llm-c-03`).

---

## 3. Market-problem-driven project portfolio

Each row = a sellable product whose runtime is 100% C; the only external dependency is an OpenAI-compatible endpoint.

### Tier A — weekend builds (reuse §2 kernel verbatim, change tools/prompt)

| # | Project | Market problem solved | Who pays | Secret sauce |
|---|---|---|---|---|
| A1 | **`agsh` — terminal copilot agent** (curl-able CLI: `agsh "find why nginx crashed last night"` → runs journalctl/grep tools itself) | Sysadmins drown in log spelunking; existing tools send data to SaaS | DevOps teams wanting on-prem, air-gapped ops | One static binary, no Python, works inside distroless containers; local-only mode via base_url to ollama |
| A2 | **Webhook→Agent glue daemon** ("Zapier but self-hosted, 40 MB RAM") | Automation platforms price by task & exfiltrate data through cloud | SMBs, homelab, IT depts | agentd + cron/webhook listener (MHD micro-httpd); tools = email, files, REST calls |
| A3 | **Offline travel/legal/medical field assistant** | Workers in tunnels/ships/mines have no connectivity; cloud agents useless | Field-service companies | Runs against local Granite-3.x-8B / Phi-4-mini via llama.cpp server; battery-friendly C runtime |
| A4 | **`pdfgrep-agent` — document Q&A appliance** | Firms hold thousands of scanned PDFs; OCR indexes are dumb keyword search | Legal, accounting, records mgmt | poppler in C extracts text → chunk → embed via small embedding model → RAG loop; ship as Raspberry-Pi-sized box |
| A5 | **Smart-home rule brain** | Home Assistant automations are brittle if/then; LLM add-ons need HA Python fork | Home-lab + smart-home vendors | Event bus (MQTT via libmosquitto) → agent decides actions; sub-1% CPU idle footprint sells it |

### Tier B — 2–6 week products (kernel + one hard subsystem)

| # | Project | Market problem | Differentiator |
|---|---|---|---|
| B1 | **Code-review CI agent** (`git diff → review → inline comments via GitHub API`) | PR review bottleneck; SaaS reviewers leak proprietary code | Binary dropped into any CI image, no pip install; runs fully against customer's own Azure-OpenAI/Groq base_url; deterministic audit log proves what was read |
| B2 | **Support-triage agent with memory** | Ticket queues waste senior engineer time; existing bots hallucinate policy | sqlite long-term memory + retrieval of past resolved tickets; guard.c enforces "never promise refunds" style policy as tool-allowlist; cost budget per ticket |
| B3 | **Voice-notes → structured CRM agent** (Whisper.cpp + agentd) | Sales reps hate CRM data entry; cloud transcription raises GDPR issues | Whisper.cpp (already C/C++) + local LLM = end-to-end offline pipeline; phone-home only when user opts in |
| B4 | **Embedded SQL analyst agent** ("ask your database") | Business users can't write SQL; BI licenses expensive | Schema introspection tool + query-execution tool with read-only guardrail + row-limit budget; ships as single binary next to Postgres |
| B5 | **Compliance evidence collector** | SOC2/ISO auditors ask for proof; teams screenshot manually for weeks | Hash-chained audit log (§2.1.6) IS the deliverable; agent runs scheduled checks via shell tools, signs transcripts; sold per-audit |
| B6 | **Edge video-event agent** (RTSP frames → small VLM via /v1 → alerts) | Retail/factories want incident detection without cloud video egress fees | ffmpeg-in-C grabs sparse frames → vision-capable hosted model (Llama 4 Vision / Qwen2.5-VL via base_url) → alert hooks; 64 MB RAM box per camera cluster |

### Tier C — quarter-long platform plays

| # | Project | Pitch |
|---|---|---|
| C1 | **`agentd` — multi-agent orchestrator daemon** (supervisor spawning child agent processes, inter-agent mailboxes over unix sockets, role graphs from TOML) | "systemd for AI agents." Fleet operators manage hundreds of agents on one host; each agent pinned to different base_url (cheap model for workers, strong model for critic). Sell to enterprises standardizing internal automation on open protocols. |
| C2 | **Speculative-decoding gateway** (small local draft model from llm.c + big cloud target via API) | Cut OpenAI bill 2–5× transparently: proxy accepts /v1, drafts tokens locally (our llm.c inference engine!), verifies against cloud in parallel batches. Latency math favors drafters at 1000+ tok/s local GPU. Unique because nobody has a C token-level proxy with a *native* drafter. |
| C3 | **Deterministic agent replay/debug service** | Record every API request/response + tool output; replay bit-exact (our mt19937/xorshift sampler docs make local replay exact). Sold as "flight recorder for agents" to teams burned by non-reproducible LangChain bugs. |
| C4 | **Air-gapped sovereign agent stack** (agentd + llm.c inference + Granite/Llama weights, shipped on encrypted SSD) | Defense, pharma, gov procurement requires zero egress. The entire loop — inference included — is C with no runtime deps; passes hardening reviews far easier than Python stacks. |
| C5 | **Tiny-model router marketplace** | GPT-2-class intent classifier (trained with train_gpt2 recipe on customer logs) routes requests to cheapest capable backend (Groq vs OpenAI vs local). Savings dashboard = revenue share on saved tokens. |
| C6 | **On-device continual-learning personal agent** | Nightly: agent's day-log → custom shard packer (llm.c data format, doc `llm-c-02`) → LoRA fine-tune (Tier-2 idea #9) → new adapter hot-swapped into local server. "The assistant that actually learns you," fully local. Long-horizon moat nobody in Python can match on low-RAM hardware. |

---

## 4. Non-negotiable features baked into the kernel (selling points)

1. **Safety rails in C, not prompts**: seccomp+rlimit sandbox around `shell` tool; per-profile tool allowlists; URL allowlists for `http_get`; max spend/session counter. Prompt-injection defense = attacker must defeat OS-level sandbox, not a paragraph in the system prompt.
2. **Auditability**: every loop iteration writes `{ts, model, base_url_hash, prompt_sha256, completion_sha256, tool, args, rc, tokens, cost}` to hash-chained JSONL → exportable compliance artifact.
3. **Model portability**: switching OpenAI↔Anthropic↔ollama↔llm.c-local = edit two lines of TOML. Benchmark harness re-runs same agent tasks across backends → publishable "which brain for which job" leaderboards (marketing flywheel).
4. **Cost visibility**: token counts come free in every response; aggregate per-agent/per-customer instantly — billable feature for agencies.

## 5. Concrete first milestone (recommendation)

Build **A1 (`agsh`) + B1 (code-review agent)** together: they share the identical kernel and prove both interactive and headless modes. Deliverables:
- `agentc/` repo: modules §2.1 (~4 kLoC total), CI that tests against mock-server (record/replay fixtures) so no API key needed in CI.
- Demo videos: (a) air-gapped laptop with ollama base_url, (b) same binary pointed at Groq key.
- Then spin C3 (replay) out of test fixtures nearly for free.

## 6. Risks & mitigations
- **SSE streaming parsing bugs** → property-test with fuzz corpus of real provider streams.
- **Provider schema drift** (tool-call formats differ subtly: OpenAI vs Ollama vs vLLM) → thin adapter layer per provider family, unit-tested against recorded fixtures.
- **Prompt injection via tool outputs** → treat all observations as untrusted string data; never eval observation text as command; sandbox covers worst case.
- **libcurl TLS trust on exotic targets** → vendored curl + ca-bundle embedded flag for appliance SKUs.
