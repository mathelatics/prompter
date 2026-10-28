/* remote.c — M3: Remote Viewer (:7373) + Director Mode (:7575).
 * Two cp_httpd servers around a live cp_engine. Viewer gets a read-only
 * word-highlight page fed by 10 Hz WebSocket pushes; director gets an editor
 * with push-to-prompter controls and the delivered-text read lock (HTTP 409).
 * QR codes for both URLs are printed to the terminal and served as SVG. */
#define _POSIX_C_SOURCE 200809L
#include "remote.h"
#include "http_ws.h"
#include <netdb.h>
#include "../vendor/qrcodegen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <arpa/inet.h>

struct cp_remote {
    cp_engine *eng;
    cp_httpd  *viewer, *director;
    int        vport, dport;
    double     last_push;      /* CLOCK_MONOTONIC seconds of last broadcast */
};

/* ---------------- tiny helpers ---------------- */

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec / 1e9;
}

/* JSON-escape src into dst (dst_cap bytes). Returns dst. */
static void jesc(char *dst, size_t cap, const char *src) {
    size_t o = 0;
    for (const unsigned char *p = (const unsigned char*)src; *p && o + 2 < cap; p++) {
        const char *rep = NULL; char buf[8];
        switch (*p) {
        case '"':  rep = "\\\""; break;
        case '\\': rep = "\\\\"; break;
        case '\n': rep = "\\n";  break;
        case '\r': rep = "";     break;
        case '\t': rep = "\\t";  break;
        default:
            if (*p < 0x20) { snprintf(buf, sizeof buf, "\\u%04x", *p); rep = buf; }
            break;
        }
        if (rep) {
            size_t l = strlen(rep);
            if (o + l >= cap) break;
            memcpy(dst + o, rep, l); o += l;
        } else dst[o++] = (char)*p;
    }
    dst[o] = '\0';
}

/* Build the shared state JSON. Includes script text when want_script != 0
 * (director stream needs it to keep its editor in sync after pushes). */
static void state_json(cp_engine *eng, int want_script, char *out, size_t cap) {
    cp_engine_state st;
    cp_engine_snapshot(eng, &st);
    char esc[64];
    jesc(esc, sizeof esc, "");   /* exercise esc path once; harmless */
    (void)esc;
    size_t o = 0;
    o += (size_t)snprintf(out + o, cap - o,
        "{\"pos\":%.3f,\"maxDelivered\":%.3f,\"speedWps\":%.3f,\"nw\":%zu,"
        "\"wordIndex\":%d,\"pageIndex\":%d,\"paused\":%d,\"speaking\":%d,"
        "\"finished\":%d,\"level\":%.3f,\"wpm\":%.1f,\"mode\":%d",
        st.pos, st.max_delivered, st.speed_wps, st.nw, st.word_index,
        st.page_index, st.paused, st.speaking, st.finished,
        st.level, st.wpm, st.mode);
    if (want_script) {
        const cp_script *s = cp_engine_script(eng);
        char *body = calloc(s->nw * 3 + 64, 1);
        if (body) {
            size_t b = 0;
            for (size_t i = 0; i < s->nw && b + 8 < cap; i++) {
                const cp_word *w = cp_word_at(s, i);
                if (!w) continue;
                char ej[512];
                jesc(ej, sizeof ej, w->text ? w->text : "");
                int n = snprintf(body + b, cap - b - 32, "%s\"%s\"",
                                 i ? "," : "", ej);
                if (n < 0 || (size_t)n > cap - b - 32) break;
                b += (size_t)n;
            }
            o += (size_t)snprintf(out + o, cap - o, ",\"words\":[%s]", body);
            free(body);
        }
    }
    snprintf(out + o, cap - o, "}");
}

static int json_get_str(const char *body, const char *key, char *out, size_t cap) {
    /* minimal scanner: finds "key":"value" honouring \" escapes */
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\":\"", key);
    const char *p = strstr(body, pat);
    if (!p) return -1;
    p += strlen(pat);
    size_t o = 0;
    while (*p && o + 1 < cap) {
        if (*p == '\\' && p[1]) { out[o++] = p[1]; p += 2; continue; }
        if (*p == '"') break;
        out[o++] = *p++;
    }
    out[o] = '\0';
    return 0;
}
static int json_get_num(const char *body, const char *key, double *val) {
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\":", key);
    const char *p = strstr(body, pat);
    if (!p) return -1;
    p += strlen(pat);
    char *end;
    *val = strtod(p, &end);
    return end == p ? -1 : 0;
}

/* ---------------- embedded web UIs ---------------- */

static const char VIEWER_HTML[] =
"<!doctype html><html><head><meta charset='utf-8'>"
"<meta name='viewport' content='width=device-width,initial-scale=1'>"
"<title>cPrompter viewer</title><style>"
"body{background:#000;color:#fff;font-family:system-ui;margin:0;padding:1em}"
"#w{font-size:11vw;text-align:center;margin-top:35vh;transition:color .1s}"
".dim{color:#555}.hot{color:#ffd23f}"
"#bar{position:fixed;top:0;left:0;height:4px;background:#ffd23f;width:0%%}"
"#st{position:fixed;bottom:.4em;right:.6em;color:#666;font-size:12px}</style></head>"
"<body><div id='bar'></div><div id='w'>connecting…</div><div id='st'></div>"
"<script>var ws,q=setInterval;"
"function url(){return 'ws://'+location.host+'/ws'}"
"function conn(){ws=new WebSocket(url());ws.onmessage=e=>show(JSON.parse(e.data));"
"ws.onclose=()=>{document.getElementById('w').textContent='reconnecting…';setTimeout(conn,1500)}}"
"function show(s){var w=document.getElementById('w');"
"w.innerHTML='<span class=dim>'+esc(s.words?s.words.slice(Math.max(0,s.wordIndex-3),s.wordIndex)"
".join(' '):'')+'</span> <span class=hot>'+esc((s.words||[''])[s.wordIndex]||'')+'</span> '"
"+'<span class=dim>'+esc(s.words?s.words.slice(s.wordIndex+1,s.wordIndex+4).join(' '):'')+'</span>';"
"document.getElementById('bar').style.width=(100*s.pos/Math.max(1,s.nw))+'%%';"
"document.getElementById('st').textContent=(s.wpm|0)+' wpm · '+(s.paused?'⏸':'▶')+(s.speaking?' 🎤':'');}"
"function esc(t){return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;')}"
"conn();</script></body></html>";

static const char DIRECTOR_HTML[] =
"<!doctype html><html><head><meta charset='utf-8'>"
"<meta name='viewport' content='width=device-width,initial-scale=1'>"
"<title>cPrompter director</title><style>"
"body{font-family:system-ui;background:#111;color:#eee;margin:0;padding:1em}"
"textarea{width:100%%;height:40vh;background:#1c1c1c;color:#eee;border:1px solid #333;"
"border-radius:6px;padding:.6em;font-size:15px;box-sizing:border-box}"
"button{margin:.3em .3em .3em 0;padding:.5em 1em;border-radius:6px;border:1px solid #444;"
"background:#222;color:#eee;cursor:pointer}button:hover{background:#333}"
"#lock{color:#f66;font-size:13px}#live{color:#8cf;font-size:14px;margin:.4em 0}"
"del{color:#666;text-decoration:none}</style></head><body>"
"<h3>🎬 cPrompter Director</h3><div id='live'>offline</div>"
"<textarea id='t' placeholder='script…'></textarea>"
"<div id='lock'></div>"
"<div><button onclick='push()'>📤 Push script</button>"
"<button onclick='post(\"pause\",{})'>⏯ Pause/Resume</button>"
"<button onclick=\"post('speed',{wps:+prompt('words/sec',3)||3})\">⚡ Speed</button>"
"<button onclick=\"post('jump',{word:+prompt('jump to word #',0)||0})\">⤴ Jump</button></div>"
"<script>var ws,words=[],maxD=0;"
"function url(){return 'ws://'+location.host+'/ws'}"
"function conn(){ws=new WebSocket(url());ws.onmessage=e=>live(JSON.parse(e.data));"
"ws.onclose=()=>{document.getElementById('live').textContent='reconnecting…';setTimeout(conn,1500)}}"
"function live(s){maxD=s.maxDelivered;"
"if(s.words&&(!words.length||document.activeElement.id!='t')){words=s.words;"
"if(document.activeElement.id!='t')document.getElementById('t').value=words.join(' ')}"
"document.getElementById('live').textContent='▶ pos '+Math.round(s.pos)+'/'+s.nw+"
"' · delivered '+Math.round(maxD)+' · '+(s.wpm|0)+' wpm'+(s.paused?' · PAUSED':'');"
"document.getElementById('lock').textContent='read-lock: first '+Math.floor(maxD)+\" words can't be edited\";}"
"function post(k,b){fetch('/api/'+k,{method:'POST',headers:{'Content-Type':'application/json'},"
"body:JSON.stringify(b)}).then(r=>r.json()).then(j=>{if(j.error)alert(j.error);if(j.ok)live(j)})}"
"function push(){post('script',{text:document.getElementById('t').value})}"
"conn();</script></body></html>";

/* ---------------- request routing ---------------- */

typedef struct { cp_remote *r; int is_director; } route_ctx;

static void respond_json(cp_http_req *req, int status, const char *json) {
    req->status = status;
    snprintf(req->ctype, req->ctype_cap, "application/json");
    size_t n = strlen(json);
    if (n >= req->out_cap) n = req->out_cap - 1;
    memcpy(req->out, json, n);
    req->out_len = n;
}

static void qr_svg(const char *url, char *out, size_t cap) {
    cp_qr *qr = cp_qr_encode(url);
    if (!qr) { snprintf(out, cap, "<svg xmlns='http://www.w3.org/2000/svg'/>"); return; }
    size_t o = 0;
    int q = 4, s = qr->size, dim = s + 2 * q;
    o += (size_t)snprintf(out + o, cap - o,
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 %d %d' shape-rendering='crispEdges'>"
        "<rect width='%d' height='%d' fill='#fff'/>", dim, dim, dim, dim);
    for (int y = 0; y < s && o + 90 < cap; y++)
        for (int x = 0; x < s && o + 90 < cap; x++)
            if (qr->pixels[y * s + x] == '#')
                o += (size_t)snprintf(out + o, cap - o,
                    "<rect x='%d' y='%d' width='1' height='1'/>", x + q, y + q);
    snprintf(out + o, cap - o, "</svg>");
    cp_qr_free(qr);
}

static void handler(cp_http_req *req) {
    route_ctx *rc = (route_ctx *)req->user;
    cp_remote *r = rc->r;
    int dir = rc->is_director;

    if (strcmp(req->path, "/ws") == 0) return;   /* http_ws handles upgrade */

    if (strcmp(req->method, "GET") == 0) {
        if (strcmp(req->path, "/") == 0) {
            const char *html = dir ? DIRECTOR_HTML : VIEWER_HTML;
            snprintf(req->ctype, req->ctype_cap, "text/html; charset=utf-8");
            size_t n = strlen(html);
            if (n < req->out_cap) { memcpy(req->out, html, n); req->out_len = n; }
            return;
        }
        if (strcmp(req->path, "/api/state") == 0) {
            char js[65536];
            state_json(r->eng, dir, js, sizeof js);
            respond_json(req, 200, js);
            return;
        }
        if (strcmp(req->path, "/qr.svg") == 0) {
            char url[128];
            snprintf(url, sizeof url, "http://%s:%d",
                     dir ? "HOST" : "HOST", r->vport);
            /* real host filled by caller env; fall back to LAN hint */
            const char *lh = getenv("CP_HOST");
            snprintf(url, sizeof url, "http://%s:%d", lh ? lh : "127.0.0.1",
                     dir ? r->dport : r->vport);
            snprintf(req->ctype, req->ctype_cap, "image/svg+xml");
            qr_svg(url, req->out, req->out_cap);
            req->out_len = strlen(req->out);
            return;
        }
        respond_json(req, 404, "{\"error\":\"not found\"}");
        return;
    }

    if (strcmp(req->method, "POST") != 0) {
        respond_json(req, 400, "{\"error\":\"method\"}"); return;
    }
    if (!dir) { respond_json(req, 403, "{\"error\":\"viewer is read-only\"}"); return; }

    const char *b = req->body;
    if (strcmp(req->path, "/api/pause") == 0) {
        cp_engine_toggle_pause(r->eng);
    } else if (strcmp(req->path, "/api/speed") == 0) {
        double wps;
        if (json_get_num(b, "wps", &wps) != 0 || wps <= 0) {
            respond_json(req, 400, "{\"error\":\"wps required\"}"); return; }
        cp_engine_set_speed(r->eng, wps);
    } else if (strcmp(req->path, "/api/jump") == 0) {
        double w;
        if (json_get_num(b, "word", &w) != 0) {
            respond_json(req, 400, "{\"error\":\"word required\"}"); return; }
        cp_engine_jump(r->eng, (size_t)w, 1);
    } else if (strcmp(req->path, "/api/script") == 0) {
        char *text = malloc(1 << 20);
        if (!text || json_get_str(b, "text", text, 1 << 20) != 0) {
            free(text);
            respond_json(req, 400, "{\"error\":\"text required\"}"); return; }
        char err[256];
        int rcv = cp_engine_apply_edit(r->eng, text, err, sizeof err);
        free(text);
        if (rcv != 0) {
            char js[512];
            char e2[256]; jesc(e2, sizeof e2, err);
            snprintf(js, sizeof js, "{\"error\":\"read-lock: %s\"}", e2);
            respond_json(req, 409, js);
            return;
        }
        cp_engine_after_edit(r->eng);
        char js[65536];
        state_json(r->eng, 1, js, sizeof js);
        cp_httpd_ws_broadcast(r->director, js);
        char js2[65536];
        state_json(r->eng, 1, js2, sizeof js2);
        cp_httpd_ws_broadcast(r->viewer, js2);
        respond_json(req, 200, js);
        return;
    } else {
        respond_json(req, 404, "{\"error\":\"not found\"}"); return;
    }
    char js[65536];
    state_json(r->eng, 1, js, sizeof js);
    cp_httpd_ws_broadcast(r->director, js);
    char js2[65536];
    state_json(r->eng, 1, js2, sizeof js2);
    cp_httpd_ws_broadcast(r->viewer, js2);
    respond_json(req, 200, js);
}

/* ---------------- public API ---------------- */

static route_ctx ctx_viewer, ctx_director;

cp_remote *cp_remote_start(cp_engine *eng, int viewer_port, int director_port) {
    cp_remote *r = calloc(1, sizeof *r);
    if (!r) return NULL;
    r->eng = eng; r->vport = viewer_port; r->dport = director_port;
    r->last_push = 0;
    ctx_viewer = (route_ctx){ r, 0 };
    ctx_director = (route_ctx){ r, 1 };
    r->viewer = cp_httpd_start(viewer_port, handler, &ctx_viewer);
    r->director = cp_httpd_start(director_port, handler, &ctx_director);
    if (!r->viewer || !r->director) { cp_remote_stop(r); return NULL; }
    return r;
}

void cp_remote_stop(cp_remote *r) {
    if (!r) return;
    if (r->viewer) cp_httpd_stop(r->viewer);
    if (r->director) cp_httpd_stop(r->director);
    free(r);
}

void cp_remote_tick(cp_remote *r) {
    if (!r) return;
    double t = now_s();
    if (t - r->last_push < 0.1) return;      /* 10 Hz cap */
    r->last_push = t;
    char js[65536];
    state_json(r->eng, 0, js, sizeof js);
    cp_httpd_ws_broadcast(r->viewer, js);
    state_json(r->eng, 1, js, sizeof js);
    cp_httpd_ws_broadcast(r->director, js);
}

size_t cp_remote_viewers(cp_remote *r)   { return r ? cp_httpd_ws_count(r->viewer) : 0; }
size_t cp_remote_directors(cp_remote *r) { return r ? cp_httpd_ws_count(r->director) : 0; }

void cp_remote_print_qr(cp_remote *r) {
    if (!r) return;
    const char *lh = getenv("CP_HOST");
    char host[64];
    if (lh && *lh) snprintf(host, sizeof host, "%s", lh);
    else if (gethostname(host, sizeof host) == 0) {
        struct hostent *he = gethostbyname(host);
        if (he && he->h_addr_list[0])
            snprintf(host, sizeof host, "%s",
                     inet_ntoa(*(struct in_addr*)he->h_addr_list[0]));
    } else snprintf(host, sizeof host, "127.0.0.1");
    char url[128];
    snprintf(url, sizeof url, "http://%s:%d", host, r->vport);
    printf("\nRemote viewer  → %s\n", url);
    cp_qr *q = cp_qr_encode(url); if (q) { cp_qr_print(q); cp_qr_free(q); }
    snprintf(url, sizeof url, "http://%s:%d", host, r->dport);
    printf("Director mode  → %s\n", url);
    q = cp_qr_encode(url); if (q) { cp_qr_print(q); cp_qr_free(q); }
    fflush(stdout);
}
