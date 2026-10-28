/* ai/openai.c — see openai.h */
#include "openai.h"
#include "http_client.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* escape a UTF-8 string into JSON string literal body (dst excludes quotes) */
static void jesc(const char *src, char *dst, size_t dn) {
    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)src; *p && o + 7 < dn; p++) {
        switch (*p) {
            case '"':  dst[o++]='\\'; dst[o++]='"'; break;
            case '\\': dst[o++]='\\'; dst[o++]='\\'; break;
            case '\n': dst[o++]='\\'; dst[o++]='n'; break;
            case '\r': dst[o++]='\\'; dst[o++]='r'; break;
            case '\t': dst[o++]='\\'; dst[o++]='t'; break;
            default:
                if (*p < 0x20) { o += (size_t)snprintf(dst+o, dn-o, "\\u%04x", *p); }
                else dst[o++] = (char)*p;
        }
    }
    dst[o] = 0;
}

int cp_openai_cfg_from_env(cp_openai_cfg *cfg) {
    memset(cfg, 0, sizeof *cfg);
    const char *b = getenv("CCP_BASE_URL");
    snprintf(cfg->base_url, sizeof cfg->base_url, "%s",
             b && *b ? b : "http://127.0.0.1:8080/v1");
    const char *m = getenv("CCP_MODEL");
    snprintf(cfg->model, sizeof cfg->model, "%s", m ? m : "");
    const char *k = getenv("CCP_API_KEY");
    if (!k || !*k) k = getenv("OPENAI_API_KEY");
    snprintf(cfg->api_key, sizeof cfg->api_key, "%s", k ? k : "");
    cfg->timeout_ms = 15000;
    /* strip trailing slash so url+/chat/completions is clean */
    size_t L = strlen(cfg->base_url);
    while (L > 0 && cfg->base_url[L-1] == '/') cfg->base_url[--L] = 0;
    if (!strstr(cfg->base_url, "://")) return -1;
    return 0;
}

/* resolve model name: if cfg->model empty, GET /models and take first id */
static void resolve_model(const cp_openai_cfg *cfg, char *model, size_t mn) {
    if (cfg->model[0]) { snprintf(model, mn, "%s", cfg->model); return; }
    char url[512];
    snprintf(url, sizeof url, "%s/models", cfg->base_url);
    cp_http_resp r;
    if (cp_http_get(url, cfg->api_key[0] ? cfg->api_key : NULL, 5000, &r) == 0
        && r.status == 200 && r.body) {
        /* data.0.id — llama.cpp/Ollama/OpenAI all use this shape */
        if (cp_json_find_path(r.body, "data.0.id", model, mn)) goto done;
        if (cp_json_get_string(r.body, "id", model, mn)) goto done;
    }
    snprintf(model, mn, "default");
done:
    cp_http_resp_free(&r);
}

int cp_openai_chat(const cp_openai_cfg *cfg, const char *system,
                   const char *user, double temperature,
                   char *out, size_t outn, char *err, size_t errn) {
    if (outn) out[0] = 0;
    char model[128];
    resolve_model(cfg, model, sizeof model);

    char *esys = malloc(strlen(system) * 6 + 3);
    char *eusr = malloc(strlen(user) * 6 + 3);
    char *body = malloc(strlen(esys) + strlen(eusr) + 512);
    if (!esys || !eusr || !body) { free(esys); free(eusr); free(body);
        snprintf(err, errn, "oom"); return -1; }
    jesc(system, esys, strlen(system) * 6 + 3);
    jesc(user, eusr, strlen(user) * 6 + 3);
    snprintf(body, strlen(esys) + strlen(eusr) + 512,
        "{\"model\":\"%s\",\"temperature\":%.3f,\"stream\":false,"
        "\"messages\":[{\"role\":\"system\",\"content\":\"%s\"},"
        "{\"role\":\"user\",\"content\":\"%s\"}]}",
        model, temperature, esys, eusr);
    free(esys); free(eusr);

    char url[512];
    snprintf(url, sizeof url, "%s/chat/completions", cfg->base_url);
    cp_http_resp r;
    int rc = cp_http_post_json(url, body,
                               cfg->api_key[0] ? cfg->api_key : NULL,
                               cfg->timeout_ms ? cfg->timeout_ms : 15000, &r);
    free(body);
    if (rc != 0) { snprintf(err, errn, "transport: %s", r.err); return -1; }
    if (r.status != 200) {
        snprintf(err, errn, "http %ld: %.180s", r.status, r.body ? r.body : "");
        cp_http_resp_free(&r); return -1;
    }
    /* OpenAI shape: choices[0].message.content ; some local servers also
     * return top-level "content" — try both. */
    if (!cp_json_find_path(r.body, "choices.0.message.content", out, outn)) {
        if (!cp_json_get_string(r.body, "content", out, outn)) {
            snprintf(err, errn, "no content in response: %.180s", r.body);
            cp_http_resp_free(&r); return -1;
        }
    }
    cp_http_resp_free(&r);
    return 0;
}
