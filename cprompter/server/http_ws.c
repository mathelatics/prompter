/* http_ws.c — hand-rolled HTTP/1.1 + WebSocket server (see http_ws.h).
 * Design: one listener thread; per-client connection threads (few clients on
 * a LAN teleprompter, so thread-per-conn is fine and keeps it simple).
 * Supports: GET/POST with Content-Length bodies, keep-alive off (Connection:
 * close), WS upgrade (RFC6455 §1.3 handshake via local SHA-1+base64), server
 * text frames (no mask on outbound per spec), inbound masked frames unmasked,
 * ping->pong, close handshake. Max 8KB request headers, 1MB body. */
#define _POSIX_C_SOURCE 200809L
#include "http_ws.h"

#include <strings.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

/* ------------------------------------------------------------------ SHA-1 */
typedef struct { uint32_t h[5]; uint64_t len; } sha1_ctx;

static uint32_t rol(uint32_t v, int b) { return (v << b) | (v >> (32 - b)); }

static void sha1_block(sha1_ctx *c, const unsigned char *p) {
    uint32_t w[80], a = c->h[0], b = c->h[1], cc = c->h[2], d = c->h[3], e = c->h[4];
    for (int i = 0; i < 16; i++)
        w[i] = (p[i*4] << 24) | (p[i*4+1] << 16) | (p[i*4+2] << 8) | p[i*4+3];
    for (int i = 16; i < 80; i++)
        w[i] = rol(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if      (i < 20) { f = (b & cc) | (~b & d);        k = 0x5A827999; }
        else if (i < 40) { f = b ^ cc ^ d;                 k = 0x6ED9EBA1; }
        else if (i < 60) { f = (b & cc) | (b & d) | (cc & d); k = 0x8F1BBCDC; }
        else             { f = b ^ cc ^ d;                 k = 0xCA62C1D6; }
        uint32_t t = rol(a, 5) + f + e + k + w[i];
        e = d; d = cc; cc = rol(b, 30); b = a; a = t;
    }
    c->h[0]+=a; c->h[1]+=b; c->h[2]+=cc; c->h[3]+=d; c->h[4]+=e;
}

static void sha1(const unsigned char *msg, size_t n, unsigned char out[20]) {
    sha1_ctx c = { {0x67452301,0xEFCDAB89,0x98BADCFE,0x10325476,0xC3D2E1F0}, 0 };
    size_t i = 0;
    unsigned char blk[64];
    for (; i + 64 <= n; i += 64) sha1_block(&c, msg + i);
    size_t rem = n - i;
    memcpy(blk, msg + i, rem);
    blk[rem] = 0x80;
    if (rem >= 56) { memset(blk + rem + 1, 0, 63 - rem); sha1_block(&c, blk); rem = 0; memset(blk, 0, 64); }
    else memset(blk + rem + 1, 0, 55 - rem);
    uint64_t bits = (uint64_t)n * 8;
    for (int j = 0; j < 8; j++) blk[56 + j] = (unsigned char)(bits >> (56 - 8*j));
    sha1_block(&c, blk);
    for (int j = 0; j < 5; j++) {
        out[j*4]   = (unsigned char)(c.h[j] >> 24);
        out[j*4+1] = (unsigned char)(c.h[j] >> 16);
        out[j*4+2] = (unsigned char)(c.h[j] >> 8);
        out[j*4+3] = (unsigned char)c.h[j];
    }
}

static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static void b64(const unsigned char *in, size_t n, char *out) {
    size_t o = 0;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = in[i] << 16;
        if (i+1 < n) v |= in[i+1] << 8;
        if (i+2 < n) v |= in[i+2];
        out[o++] = B64[(v >> 18) & 63];
        out[o++] = B64[(v >> 12) & 63];
        out[o++] = i+1 < n ? B64[(v >> 6) & 63] : '=';
        out[o++] = i+2 < n ? B64[v & 63] : '=';
    }
    out[o] = 0;
}

/* ------------------------------------------------------------- conn / ws */
struct cp_ws {
    int fd;
    int open;
    pthread_mutex_t sendmu;
    struct cp_ws *next;
};

struct cp_httpd {
    int port, lfd, quit;
    pthread_t accept_thr;
    cp_http_handler h;
    cp_ws_open_cb ocb;
    void *user;
    pthread_mutex_t listmu;
    struct cp_ws *conns;
};

static int read_line(int fd, char *buf, size_t cap) {   /* reads CRLF line incl. */
    size_t n = 0;
    while (n + 2 < cap) {
        char c;
        ssize_t r = recv(fd, &c, 1, 0);
        if (r <= 0) return -1;
        buf[n++] = c;
        if (c == '\n') break;
    }
    buf[n] = 0;
    return (int)n;
}

static int hdr_get(char *hdrs, const char *name, char *out, size_t cap) {
    /* case-insensitive scan over raw header block */
    size_t nl = strlen(name);
    char *p = hdrs;
    while (*p) {
        if (!strncasecmp(p, name, nl) && p[nl] == ':') {
            char *v = p + nl + 1;
            while (*v == ' ' || *v == '\t') v++;
            char *e = strstr(v, "\r\n"); if (!e) e = v + strlen(v);
            size_t len = (size_t)(e - v);
            if (len >= cap) len = cap - 1;
            memcpy(out, v, len); out[len] = 0;
            return 0;
        }
        p = strchr(p, '\n'); if (!p) break; p++;
    }
    return -1;
}

static int send_all(int fd, const void *buf, size_t n) {
    const char *p = buf;
    while (n) { ssize_t w = send(fd, p, n, MSG_NOSIGNAL); if (w <= 0) return -1; p += w; n -= (size_t)w; }
    return 0;
}

void cp_ws_send_text(cp_ws *w, const char *text) {
    if (!w || !w->open) return;
    size_t len = strlen(text);
    unsigned char hdr[10]; size_t hl = 0;
    hdr[0] = 0x81;                                   /* FIN + text frame     */
    if (len < 126) { hdr[1] = (unsigned char)len; hl = 2; }
    else if (len < 65536) { hdr[1] = 126; hdr[2] = len >> 8; hdr[3] = len & 255; hl = 4; }
    else { hdr[1] = 127;
           for (int i = 0; i < 8; i++) hdr[2+i] = (unsigned char)((uint64_t)len >> (56 - 8*i));
           hl = 10; }
    pthread_mutex_lock(&w->sendmu);
    int rc = send_all(w->fd, hdr, hl) || (len && send_all(w->fd, text, len));
    pthread_mutex_unlock(&w->sendmu);
    if (rc) w->open = 0;
}

int cp_ws_is_open(cp_ws *w) { return w && w->open; }

static void ws_conn_remove(cp_httpd *d, cp_ws *w) {
    pthread_mutex_lock(&d->listmu);
    cp_ws **pp = &d->conns;
    while (*pp && *pp != w) pp = &(*pp)->next;
    if (*pp) *pp = w->next;
    pthread_mutex_unlock(&d->listmu);
}

static void *conn_main(void *arg) {
    struct { cp_httpd *d; int fd; } *a = arg;
    cp_httpd *d = a->d; int fd = a->fd; free(a);

    /* --- read request headers ------------------------------------------- */
    char hdrbuf[8192]; size_t hn = 0;
    char line[1024];
    /* request line */
    if (read_line(fd, line, sizeof line) < 0) goto done;
    char method[8] = "", target[1024] = "", version[16] = "";
    sscanf(line, "%7s %1023s %15s", method, target, version);
    hn += (size_t)snprintf(hdrbuf + hn, sizeof hdrbuf - hn, "%s", line);
    /* headers until blank line */
    char *hp = hdrbuf + hn;
    while (read_line(fd, line, sizeof line) > 0) {
        size_t l = strlen(line);
        if (l == 0 || (line[0] == '\r' && line[1] == '\n') || line[0] == '\n') break;
        if (hn + l + 1 < sizeof hdrbuf - 1) { strcpy(hp, line); hp += l; hn += l; *hp++ = '\n'; *hp = 0; }
    }

    {
        /* --- dispatch --------------------------------------------------- */
        static const char *WS_GUIDE = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
        char key[128] = "", upg[64] = "", clen[32] = "", ctype[128] = "";
        hdr_get(hdrbuf, "Sec-WebSocket-Key", key, sizeof key);
        hdr_get(hdrbuf, "Upgrade", upg, sizeof upg);
        hdr_get(hdrbuf, "Content-Length", clen, sizeof clen);
        hdr_get(hdrbuf, "Content-Type", ctype, sizeof ctype);

        char path[1024], query[1024] = "";
        snprintf(path, sizeof path, "%s", target);
        char *q = strchr(path, '?');
        if (q) { *q = 0; snprintf(query, sizeof query, "%s", q + 1); }

        /* body */
        size_t blen = (size_t)atol(clen);
        if (blen > 1u<<20) blen = 1u<<20;
        char *body = malloc(blen + 1);
        size_t got = 0;
        while (got < blen) {
            ssize_t r = recv(fd, body + got, blen - got, 0);
            if (r <= 0) break;
            got += (size_t)r;
        }
        if (body) body[got] = 0;

        int want_ws = key[0] && !strcasecmp(upg, "websocket");

        char outbuf[65536];
        char ctype_resp[64] = "text/plain; charset=utf-8";
        cp_http_req r = {0};
        r.method = method; r.path = path; r.query = query;
        r.body = body ? body : ""; r.body_len = got;
        r.content_type = ctype;
        r.status = 200;
        r.ctype = ctype_resp; r.ctype_cap = sizeof ctype_resp;
        r.out = outbuf; r.out_cap = sizeof outbuf - 1; r.out_len = 0;
        r.user = d->user; r.want_ws = want_ws;

        if (d->h) d->h(&r);

        if (want_ws && r.status == 200) {
            /* RFC6455 handshake */
            char cat[256]; snprintf(cat, sizeof cat, "%s%s", key, WS_GUIDE);
            unsigned char dig[20]; sha1((const unsigned char *)cat, strlen(cat), dig);
            char acc[64]; b64(dig, 20, acc);
            char resp[512];
            int rl = snprintf(resp, sizeof resp,
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                "Sec-WebSocket-Accept: %s\r\n\r\n", acc);
            if (send_all(fd, resp, (size_t)rl) == 0) {
                cp_ws *w = calloc(1, sizeof *w);
                w->fd = fd; w->open = 1;
                pthread_mutex_init(&w->sendmu, NULL);
                pthread_mutex_lock(&d->listmu);
                w->next = d->conns; d->conns = w;
                pthread_mutex_unlock(&d->listmu);
                if (d->ocb) d->ocb(w, &r, d->user);
                /* pump client frames (mask/unmask) until close/error */
                unsigned char fb[4096];
                while (w->open && !d->quit) {
                    ssize_t n = recv(fd, fb, sizeof fb, 0);
                    if (n <= 0) break;
                    for (ssize_t i = 0; i < n;) {
                        if (i + 2 > n) break;
                        unsigned char op = fb[i] & 0x0f;
                        size_t flen = fb[i+1] & 0x7f;
                        size_t need = 2;
                        if (flen == 126) { if (i+4 > n) break; flen = (fb[i+2]<<8)|fb[i+3]; need = 4; }
                        else if (flen == 127) { if (i+10 > n) break; flen = 0;
                                    for (int j=0;j<8;j++) flen = (flen<<8)|fb[i+2+j]; need = 10; }
                        if (flen > 4096) { w->open = 0; break; }
                        if (i + (ssize_t)(need + flen) > n) { i = n; break; } /* partial; skip for now */
                        if (op == 0x8) { w->open = 0; }          /* client close */
                        else if (op == 0x9) {                    /* ping -> pong */
                            unsigned char pong[140] = {0x8a, (unsigned char)flen};
                            memcpy(pong + 2, fb + i + need + 4, flen > 138 ? 138 : flen);
                            send_all(fd, pong, 2 + flen);
                        }
                        i += need + 4 + flen;                    /* +4 = mask key */
                    }
                }
                /* send close frame best-effort */
                unsigned char cl[2] = {0x88, 0};
                pthread_mutex_lock(&w->sendmu);
                send_all(fd, cl, 2);
                pthread_mutex_unlock(&w->sendmu);
                ws_conn_remove(d, w);
                pthread_mutex_destroy(&w->sendmu);
                free(w);
                goto done_body;
            }
        } else {
            /* plain HTTP response */
            const char *status_txt = r.status == 200 ? "OK" : r.status == 400 ? "Bad Request" :
                r.status == 404 ? "Not Found" : r.status == 409 ? "Conflict" :
                r.status == 405 ? "Method Not Allowed" : "Internal Server Error";
            char head[512];
            int hl = snprintf(head, sizeof head,
                "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
                r.status ? r.status : 200, status_txt, ctype_resp, r.out_len);
            send_all(fd, head, (size_t)hl);
            if (r.out_len) send_all(fd, outbuf, r.out_len);
        }
    done_body:
        free(body);
    }
done:
    close(fd);
    return NULL;
}

static void *accept_main(void *arg) {
    cp_httpd *d = arg;
    while (!d->quit) {
        struct sockaddr_in sa; socklen_t sl = sizeof sa;
        int cfd = accept(d->lfd, (struct sockaddr *)&sa, &sl);
        if (cfd < 0) { if (errno == EINTR || d->quit) break; usleep(50*1000); continue; }
        int one = 1; setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        struct timeval tv = { .tv_sec = 30, .tv_usec = 0 };
        setsockopt(cfd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        void *box = calloc(1, sizeof(struct { cp_httpd *d; int fd; }));
        ((struct { cp_httpd *d; int fd; }*)box)->d = d;
        ((struct { cp_httpd *d; int fd; }*)box)->fd = cfd;
        pthread_t th;
        if (pthread_create(&th, NULL, conn_main, box) == 0) pthread_detach(th);
        else { free(box); close(cfd); }
    }
    return NULL;
}

cp_httpd *cp_httpd_start(int port, cp_http_handler h, void *user) {
    cp_httpd *d = calloc(1, sizeof *d);
    d->port = port; d->h = h; d->user = user;
    pthread_mutex_init(&d->listmu, NULL);
    d->lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (d->lfd < 0) { free(d); return NULL; }
    int one = 1;
    setsockopt(d->lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in sa = {0};
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = INADDR_ANY;
    sa.sin_port = htons((uint16_t)port);
    if (bind(d->lfd, (struct sockaddr *)&sa, sizeof sa) || listen(d->lfd, 16)) {
        close(d->lfd); free(d); return NULL;
    }
    if (pthread_create(&d->accept_thr, NULL, accept_main, d) != 0) {
        close(d->lfd); free(d); return NULL;
    }
    return d;
}

void cp_httpd_stop(cp_httpd *d) {
    if (!d) return;
    d->quit = 1;
    shutdown(d->lfd, SHUT_RDWR);
    close(d->lfd);
    pthread_join(d->accept_thr, NULL);
    /* force-close live ws conns */
    pthread_mutex_lock(&d->listmu);
    for (cp_ws *w = d->conns; w; ) { cp_ws *nx = w->next; w->open = 0; close(w->fd); w = nx; }
    d->conns = NULL;
    pthread_mutex_unlock(&d->listmu);
    pthread_mutex_destroy(&d->listmu);
    free(d);
}

size_t cp_httpd_ws_count(cp_httpd *d) {
    if (!d) return 0;
    pthread_mutex_lock(&d->listmu);
    size_t n = 0;
    for (cp_ws *w = d->conns; w; w = w->next) if (w->open) n++;
    pthread_mutex_unlock(&d->listmu);
    return n;
}

void cp_httpd_ws_broadcast(cp_httpd *d, const char *text) {
    if (!d) return;
    pthread_mutex_lock(&d->listmu);
    for (cp_ws *w = d->conns; w; w = w->next)
        if (w->open) cp_ws_send_text(w, text);
    pthread_mutex_unlock(&d->listmu);
}

void cp_httpd_on_ws_open(cp_httpd *d, cp_ws_open_cb cb) { if (d) d->ocb = cb; }
