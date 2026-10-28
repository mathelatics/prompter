/* ai/openai — OpenAI-compatible chat-completions client (M5 coach).
 * base_url + token exactly like docs/audit/agent-c-02-reference-designs.md:
 * local llama.cpp/Ollama/our engine or any cloud, one config change. */
#ifndef CP_OPENAI_H
#define CP_OPENAI_H

#include <stddef.h>

typedef struct {
    char base_url[256];   /* e.g. http://127.0.0.1:8080/v1 or https://api.openai.com/v1 */
    char model[128];      /* served model name; "" -> query /models and use first     */
    char api_key[256];    /* bearer token; "" -> no auth header (local servers)        */
    int  timeout_ms;      /* per-request deadline, default 15000                        */
} cp_openai_cfg;

/* Fill cfg from env vars CCP_BASE_URL / CCP_MODEL / CCP_API_KEY with
 * defaults (http://127.0.0.1:8080/v1). Returns 0 ok, -1 bad url. */
int cp_openai_cfg_from_env(cp_openai_cfg *cfg);

/* POST a single-turn chat completion. system+user strings, returns assistant
 * content in out (malloc'd by caller's buffer). temperature 0..1.
 * Returns 0 on success, -1 on error (msg in err). */
int cp_openai_chat(const cp_openai_cfg *cfg, const char *system,
                   const char *user, double temperature,
                   char *out, size_t outn, char *err, size_t errn);

#endif
