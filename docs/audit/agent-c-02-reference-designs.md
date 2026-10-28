# Agent-C Reference Designs — Wire Formats, Skeletons & Model Roster (2026)

Companion to `agent-c-01-blueprint.md`. Implementation-ready details so no re-research is needed.

---

## 1. OpenAI-compatible wire protocol subset to implement

### 1.1 Chat completion (blocking)
```
POST {base_url}/chat/completions
Headers:
  Content-Type: application/json
  Authorization: Bearer {token}        # omit for local llama.cpp/ollama/vLLM defaults
Body:
{
  "model": "granite3.1-dense:8b",
  "messages": [
    {"role":"system","content":"..."},
    {"role":"user","content":"..."},
    {"role":"assistant","content":"...","tool_calls":[...]},
    {"role":"tool","tool_call_id":"call_abc","content":"observation text"}
  ],
  "tools":[{"type":"function","function":{"name":"shell","description":"...",
            "parameters":{ /* JSON Schema */ }}}],
  "tool_choice":"auto",
  "temperature":0.2,"max_tokens":1024,
  "stream":false
}
Response (key fields):
  choices[0].message.content            -> final answer OR reasoning preamble
  choices[0].message.tool_calls[]       -> {id, type:"function",
                                           function:{name, arguments(JSON string)}}
  choices[0].finish_reason              -> "stop" | "tool_calls" | "length"
  usage.{prompt_tokens,completion_tokens,total_tokens}   -> cost accounting
```

### 1.2 Streaming (SSE)
Same request with `"stream":true`. Response body is lines of:
```
data: {"choices":[{"delta":{"content":"tok"},"index":0}], ...}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",
        "function":{"name":"shell","arguments":"{\"cmd\":\"ls"}}]}}]}
data: [DONE]
```
C impl notes:
- libcurl `CURLOPT_WRITEFUNCTION`; buffer partial lines, split on `\n`, strip `data: ` prefix.
- **Tool-call argument streaming is fragmented arbitrarily** — accumulate `function.arguments` strings per `index` until `finish_reason:"tool_calls"`, then parse the concatenated JSON once. This is the #1 interop bug source across providers.
- Some providers (vLLM, Ollama) emit `finish_reason` only in a trailing chunk; some send `usage` only if `"stream_options":{"include_usage":true}` requested.

### 1.3 Embeddings
```
POST {base_url}/embeddings   {"model":"nomic-embed-text","input":["text",...]}
-> data[i].embedding : float array (768/1024/2048 dims typical)
```
Brute-force cosine in C over sqlite BLOB-stored vectors is fine ≤100k rows (memory.c design).

### 1.4 Provider quirks adapter table
| Provider family | Quirks to handle |
|---|---|
| OpenAI / Azure | `developer` role replaces `system` on new models; parallel tool calls standard; strict JSON-schema mode optional |
| Anthropic-compat gateways (OpenRouter etc.) | sometimes return `reasoning_content` extra field — ignore unknown fields, never error |
| vLLM | tool parsing needs `--enable-auto-tool-choice --tool-call-parser hermes` server-side; our client must surface HTTP 400 hints in logs |
| Ollama `/v1` | may omit `id` in tool_calls → synthesize `call_%d` |
| llama.cpp server | supports tools but finish_reason strings vary (`stop` vs `tool_calls` inconsistency); treat presence of tool_calls as authoritative over finish_reason |
| Groq/Fireworks/Together | fast but occasionally truncate at `max_tokens` mid-tool-call-json → detect unbalanced braces, retry with +tokens |

Rule: **parse defensively with jsmn token walk; never assume field order or presence.**

---

## 2. The agent loop — C skeleton (~150 LoC core)

```c
typedef struct { const char *name; const char *schema_json;
                 int (*run)(const char *args_json, char *out, size_t outmax); } Tool;

typedef struct {
  LlmCfg llm;            // base_url, api_key, model, timeout   <-- THE ONLY cloud/local switch
  Tool  *tools; int ntools;
  Mem   *mem;            // sqlite history + embeddings
  Audit *audit;          // hash-chained JSONL
  Guard *guard;          // allowlists, budget
  int max_steps; double budget_usd;
} Agent;

Result agent_run(Agent *a, const char *user_msg) {
  mem_append_user(a->mem, user_msg);
  for (int step = 0; step < a->max_steps; step++) {
    Req r = build_request(a->mem, a->tools);           // system+history+tools
    Resp resp = llm_chat(&a->llm, &r);                  // llm_client.c, one libcurl call
    audit_record(a->audit, &r, &resp);                 // sha256 chain, token usage, cost
    guard_check_cost(a->guard, resp.usage);            // hard stop if over budget
    if (resp.finish == FINISH_STOP && !resp.tool_calls) {
      mem_append_assistant(a->mem, resp.content);
      return (Result){OK, resp.content};
    }
    for each tc in resp.tool_calls:                    // usually 1, allow N parallel
      args = guard_check_tool(a->guard, tc.name, tc.args_json);  // allowlist + injection screen
      if (!args.ok) { obs = "REFUSED: policy"; }
      else          { obs = tool_dispatch(a->tools, tc.name, tc.args_json); }
      sandbox_exec(...)                                // shell tool: fork+exec under seccomp+rlimit
      mem_append_tool_result(a->mem, tc.id, obs);      // observation = UNTRUSTED string only
    }
  }
  return (Result){MAX_STEPS, last_content};
}
```

Key invariants (copy into code comments):
1. Tool output NEVER concatenated into executable context — only into `role:"tool"` messages.
2. Every `llm_chat` + every tool exec logged before execution returns (crash-safe audit).
3. Budget and step caps enforced client-side, never trust model self-report.

---

## 3. Sandbox for the `shell` tool (the part reviewers will grill)

```
fork() child:
  prctl(PR_SET_NO_NEW_PRIVS)
  seccomp BPF: allow read/write/openat/close/execve/fork/stat/lseek/mmap... ; ERRNO everything else
  setrlimit: AS 512MB, CPU 20s, FSIZE 50MB, NOFILE 256, NPROC 64
  chroot/docker alternative: pivot_root into /var/agent-sandbox (bind-mounted allowlisted dirs)
  drop to uid 65534 (nobody)
parent: pipe stdout/stderr (cap 1MB, ring-buffer tail), waitpid with timeout, kill -9 group on expiry.
```
On macOS/Windows fallback: capability-limited subprocess + path-jail via openat2(RESOLVE_IN_ROOT) equivalent or plain cwd jail + realpath prefix checks (document weaker guarantee).

---

## 4. Recommended small-model roster (2026) for base_url deployments

| Role | Models (all servable by llama.cpp/Ollama/vLLM behind /v1) | Why |
|---|---|---|
| Local brain (default) | Llama 3.1/3.2 8B-Instruct, Qwen2.5-7B/14B-Instruct | Strong tool-calling after prompt-tuning; ubiquitous GGUF support |
| Enterprise-friendly permissive | **IBM Granite 3.x 8B Instruct** | Apache 2.0, built-in thinking-mode toggle, hallucination flagging; great default for corporate SKUs |
| Tiny/fast router | Phi-4-mini (3.8B), Gemma 3 4B-it, Llama 3.2 3B | Intent classification / model routing (C5 project), sub-second on laptops |
| Safety judge | Llama Guard 3 (1B/8B), Granite Guardian 3.x | Second-pass content/tool-call screening in guard.c |
| Embeddings | nomic-embed-text-v1.5, Granite-3.x-Embed, bge-m3 | memory.c vector recall; all have OpenAI-compatible /v1/embeddings |
| Vision edge agent | Qwen2.5-VL 7B, Llama 4 Vision (hosted) | B6 RTSP frame analysis |
| Code specialist | Qwen2.5-Coder 7B/14B, StarCoder2, Granite-DocCode | B1 review agent local mode |
| Fully-local-from-our-code | GPT-2 124M distilled task heads trained via train_gpt2 recipe | routers/classifiers where we own every bit (see docs llm-c-03..05) |

Interop tip: attach `--tool-call-parser hermes` (vLLM) or use chat templates with native tool tokens; for models without native tool training, fall back to **JSON-block prompting** (` ```json {"tool":...}``` ) parsed by the same dispatch table — keep both paths in loop.c behind one `parse_action(resp)` function.

---

## 5. Repo layout to start today

```
agentc/
  src/ llm_client.[ch]  sse.[ch]  jsonwalk.[ch](jsmn)  loop.c  tools.c  tools_shell.c
       sandbox.c  memory.c  guard.c  audit.c  config.c  main.c
  vendor/ curl/ jsmn/ ini.h sqlite3.c mhd/        (all single-file-ish, license-checked)
  profiles/ agsh.toml  review.toml  triage.toml   (per-product = profile, not fork)
  tests/ fixtures/*.jsonl (recorded provider streams)  mock_server.c  replay_tests
  products/
     agsh/          -> A1  (thin main + sysadmin toolset)
     reviewbot/     -> B1  (github_api.c + diff tools)
     triaged/       -> B2  (IMAP/webhook ingest + ticket tools)
```
One kernel, many products = each "project" in the portfolio is a profile file + a tool module, which is why Tier A builds take a weekend.

## 6. Decision log / open questions
- [ ] Vendored curl vs system curl for appliance SKU (static musl build recommended)
- [ ] Memory: sqlite vs lmdb (start sqlite; lmdb only if write-heavy multi-agent)
- [ ] Multi-agent (C1): unix-socket mailboxes vs shared-memory ring; decide after single-agent fleet experience
- [ ] Whether to expose `/v1` server side too (agents-as-a-service) — defer; clients-first.
