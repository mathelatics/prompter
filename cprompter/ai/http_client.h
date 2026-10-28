/* ai/http_client — minimal blocking HTTP/1.1 client over POSIX sockets.
 * Purpose-built for OpenAI-compatible chat-completions calls (M5 coach).
 * No keep-alive, no TLS here: https:// is delegated to an optional curl
 * fallback so we never hand-roll a crypto stack. http:// works natively
 * (covers llama.cpp / Ollama / our llm.c engine on localhost and LAN). */
#ifndef CP_HTTP_CLIENT_H
#define CP_HTTP_CLIENT_H

#include <stddef.h>

typedef struct {
    long   status;        /* HTTP status, 0 = transport error          */
    char  *body;          /* malloc'd NUL-terminated response body     */
    size_t body_len;
    char   err[256];      /* populated when status == 0                */
} cp_http_resp;

void cp_http_resp_free(cp_http_resp *r);

/* POST body with Content-Type: application/json. timeout_ms bounds the
 * whole exchange (connect + read). Returns 0 on success (status may be
 * any HTTP code), -1 on transport failure (see r->err). */
int cp_http_post_json(const char *url, const char *json_body,
                      const char *bearer_token, int timeout_ms,
                      cp_http_resp *r);

/* GET, same contract. Used for /v1/models probing in tests. */
int cp_http_get(const char *url, const char *bearer_token,
                int timeout_ms, cp_http_resp *r);

#endif
