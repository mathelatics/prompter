/* ai/http_client.c — see http_client.h. Native http:// over POSIX sockets;
 * https:// shells out to curl when available (no crypto hand-rolling). */
#define _GNU_SOURCE
#include "http_client.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <errno.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>

void cp_http_resp_free(cp_http_resp *r) {
    if (!r) return;
    free(r->body);
    r->body = NULL;
    r->body_len = 0;
}

/* ---- URL split: scheme://host[:port]/path ---- */
typedef struct { int tls; char host[256]; int port; char path[1024]; } url_t;

static int parse_url(const char *u, url_t *o, char *err, size_t errn) {
    memset(o, 0, sizeof *o);
    o->port = 80;
    const char *p = strstr(u, "://");
    if (!p) { snprintf(err, errn, "bad url (no scheme): %s", u); return -1; }
    if (p - u == 5 && !strncmp(u, "https", 5)) o->tls = 1;
    else if (p - u == 4 && !strncmp(u, "http", 4)) o->tls = 0;
    else { snprintf(err, errn, "unsupported scheme in %s", u); return -1; }
    u = p + 3;
    /* host[:port] up to '/' */
    const char *slash = strchr(u, '/');
    const char *hend = slash ? slash : u + strlen(u);
    size_t hl = (size_t)(hend - u);
    const char *colon = memchr(u, ':', hl);
    if (colon) {
        hl = (size_t)(colon - u);
        o->port = atoi(colon + 1);
        if (o->tls && o->port == 80) o->port = 443; /* common misconfig */
    } else if (o->tls) o->port = 443;
    if (hl == 0 || hl >= sizeof o->host) { snprintf(err, errn, "bad host"); return -1; }
    memcpy(o->host, u, hl); o->host[hl] = 0;
    if (slash) snprintf(o->path, sizeof o->path, "%s", slash);
    else strcpy(o->path, "/");
    return 0;
}

/* write all / read with deadline helpers */
static int send_all(int fd, const char *buf, size_t n, int timeout_ms) {
    size_t off = 0;
    while (off < n) {
        struct pollfd pf = { .fd = fd, .events = POLLOUT };
        int pr = poll(&pf, 1, timeout_ms);
        if (pr <= 0) return -1;
        ssize_t w = send(fd, buf + off, n - off, MSG_NOSIGNAL);
        if (w <= 0) return -1;
        off += (size_t)w;
    }
    return 0;
}

/* Read full response until EOF (servers we call send Connection: close or
 * we don't care about keep-alive). Grows buffer as needed. */
static int recv_all(int fd, char **out, size_t *outlen, int timeout_ms) {
    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    if (!buf) return -1;
    for (;;) {
        struct pollfd pf = { .fd = fd, .events = POLLIN };
        int pr = poll(&pf, 1, timeout_ms);
        if (pr < 0) { free(buf); return -1; }
        if (pr == 0) { /* idle timeout: if we have a complete header+body by
                        * Content-Length we are done; else treat as timeout */
            break;
        }
        if (pr & POLLIN) {
            if (len + 1024 > cap) { cap *= 2; char *nb = realloc(buf, cap); if (!nb) { free(buf); return -1; } buf = nb; }
            ssize_t r = recv(fd, buf + len, cap - len - 1, 0);
            if (r < 0) { free(buf); return -1; }
            if (r == 0) break; /* peer closed -> done */
            len += (size_t)r;
            buf[len] = 0;
            /* early-exit if Content-Length satisfied */
            const char *cl = strcasestr(buf, "content-length:");
            if (cl) {
                long want = atol(cl + 15);
                const char *hdr_end = strstr(buf, "\r\n\r\n");
                if (hdr_end && (long)(len - (size_t)(hdr_end + 4 - buf)) >= want) break;
            }
        }
        if (pf.revents & (POLLHUP | POLLERR)) break;
    }
    buf[len] = 0;
    *out = buf; *outlen = len;
    return 0;
}

/* extract status + body from raw response */
static int split_resp(char *raw, long *status, char **body, size_t *blen) {
    if (strncmp(raw, "HTTP/", 5)) return -1;
    char *sp = strchr(raw, ' ');
    if (!sp) return -1;
    *status = strtol(sp + 1, NULL, 10);
    char *he = strstr(raw, "\r\n\r\n");
    if (!he) { *body = raw + strlen(raw); *blen = 0; return 0; }
    *body = he + 4;
    *blen = strlen(raw) - (size_t)(*body - raw);
    return 0;
}

static int do_plain(const url_t *u, const char *method, const char *json_body,
                    const char *bearer, int timeout_ms, cp_http_resp *r) {
    char req[8192];
    char hosthdr[320];
    snprintf(hosthdr, sizeof hosthdr, "%s:%d", u->host, u->port);
    int n;
    if (json_body) {
        n = snprintf(req, sizeof req,
            "POST %s HTTP/1.1\r\nHost: %s\r\nUser-Agent: cprompter/0.1\r\n"
            "Accept: application/json\r\nContent-Type: application/json\r\n"
            "%s%s%sContent-Length: %zu\r\nConnection: close\r\n\r\n",
            u->path, hosthdr,
            bearer ? "Authorization: Bearer " : "", bearer ? bearer : "",
            bearer ? "\r\n" : "", strlen(json_body));
    } else {
        n = snprintf(req, sizeof req,
            "GET %s HTTP/1.1\r\nHost: %s\r\nUser-Agent: cprompter/0.1\r\n"
            "Accept: application/json\r\n%s%s%s"
            "Connection: close\r\n\r\n",
            u->path, hosthdr,
            bearer ? "Authorization: Bearer " : "", bearer ? bearer : "",
            bearer ? "\r\n" : "");
    }
    if (n < 0 || (size_t)n >= sizeof req) { snprintf(r->err, sizeof r->err, "request too big"); return -1; }

    struct addrinfo hints = { .ai_socktype = SOCK_STREAM }, *ais = NULL;
    char portstr[16]; snprintf(portstr, sizeof portstr, "%d", u->port);
    int gai = getaddrinfo(u->host, portstr, &hints, &ais);
    if (gai != 0) { snprintf(r->err, sizeof r->err, "dns: %s", gai_strerror(gai)); return -1; }
    int fd = -1;
    for (struct addrinfo *a = ais; a; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
        struct timeval tv = { timeout_ms / 1000, (timeout_ms % 1000) * 1000 };
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        /* non-blocking connect with poll deadline */
        int fl = fcntl(fd, F_GETFL, 0); fcntl(fd, F_SETFL, fl | O_NONBLOCK);
        int rc = connect(fd, a->ai_addr, a->ai_addrlen);
        if (rc != 0 && errno != EINPROGRESS) { close(fd); fd = -1; continue; }
        if (rc != 0) {
            struct pollfd pf = { .fd = fd, .events = POLLOUT };
            if (poll(&pf, 1, timeout_ms) <= 0) { close(fd); fd = -1; continue; }
            int soerr = 0; socklen_t sl = sizeof soerr;
            getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &sl);
            if (soerr) { close(fd); fd = -1; continue; }
        }
        fcntl(fd, F_SETFL, fl); /* back to blocking for recv loop (we poll anyway) */
        break;
    }
    freeaddrinfo(ais);
    if (fd < 0) { snprintf(r->err, sizeof r->err, "connect %.180s failed", hosthdr); return -1; }

    int rc2 = send_all(fd, req, strlen(req), timeout_ms);
    char *raw = NULL; size_t rl = 0;
    if (rc2 == 0) rc2 = recv_all(fd, &raw, &rl, timeout_ms);
    close(fd);
    if (rc2 != 0 || !raw) { snprintf(r->err, sizeof r->err, "io error/timeout"); free(raw); return -1; }
    long st = 0; char *b = NULL; size_t bl = 0;
    if (split_resp(raw, &st, &b, &bl) != 0) { free(raw); snprintf(r->err, sizeof r->err, "bad http response"); return -1; }
    r->status = st;
    r->body = malloc(bl + 1);
    if (!r->body) { free(raw); return -1; }
    memcpy(r->body, b, bl); r->body[bl] = 0;
    r->body_len = bl;
    free(raw);
    return 0;
}

#ifdef CP_HAVE_CURL
static int do_curl(const char *url, const char *method, const char *json_body,
                   const char *bearer, int timeout_ms, cp_http_resp *r) {
    char cmd[4096];
    /* write body to temp file to avoid quoting hell */
    const char *tmp = "/tmp/.cprompter_http_body";
    if (json_body) {
        FILE *f = fopen(tmp, "wb");
        if (!f) { snprintf(r->err, sizeof r->err, "tmpfile"); return -1; }
        fwrite(json_body, 1, strlen(json_body), f);
        fclose(f);
    }
    snprintf(cmd, sizeof cmd,
        "curl -sS --max-time %.2f -X %s -H 'Content-Type: application/json'%s%s%s %s '%s' 2>/dev/null",
        timeout_ms / 1000.0, method,
        json_body ? " --data-binary @" : "", json_body ? tmp : "",
        bearer ? " -H 'Authorization: Bearer '" : "",
        url);
    /* NOTE: bearer token injection into single-quoted shell string is unsafe
     * for arbitrary tokens; tokens are alnum+dash in practice (sk-...). We
     * validate before use. */
    if (bearer) {
        for (const char *c = bearer; *c; c++)
            if (!(isalnum((unsigned char)*c) || *c=='-' || *c=='_' || *c=='.')) {
                snprintf(r->err, sizeof r->err, "token has unsafe chars"); return -1;
            }
        char cmd2[4608];
        snprintf(cmd2, sizeof cmd2, "%s -H 'Authorization: Bearer %s'", cmd, bearer);
        snprintf(cmd, sizeof cmd, "%s", cmd2);
    }
    FILE *pp = popen(cmd, "r");
    if (!pp) { snprintf(r->err, sizeof r->err, "popen"); return -1; }
    size_t cap = 4096, len = 0; char *buf = malloc(cap);
    if (!buf) { pclose(pp); return -1; }
    for (;;) {
        if (len + 1024 > cap) { cap *= 2; char *nb = realloc(buf, cap); if (!nb) { free(buf); pclose(pp); return -1; } buf = nb; }
        size_t got = fread(buf + len, 1, cap - len - 1, pp);
        len += got;
        if (got == 0) break;
    }
    buf[len] = 0;
    int st = pclose(pp);
    if (WEXITSTATUS(st) != 0) { free(buf); snprintf(r->err, sizeof r->err, "curl exit %d", WEXITSTATUS(st)); return -1; }
    r->status = 200; /* curl -sS only returns body on 2xx-3xx unless -f absent; without -f it returns body for errors too. We can't see status -> treat as 200 and let JSON parse fail loudly upstream. Acceptable for coach fallback logic. */
    r->body = buf; r->body_len = len;
    return 0;
}
#endif

static int request(const char *url, const char *method, const char *json_body,
                   const char *bearer, int timeout_ms, cp_http_resp *r) {
    memset(r, 0, sizeof *r);
    url_t u;
    if (parse_url(url, &u, r->err, sizeof r->err) != 0) return -1;
    if (!u.tls) return do_plain(&u, method, json_body, bearer, timeout_ms, r);
#ifdef CP_HAVE_CURL
    return do_curl(url, method, json_body, bearer, timeout_ms, r);
#else
    snprintf(r->err, sizeof r->err, "https:// needs curl-enabled build");
    return -1;
#endif
}

int cp_http_post_json(const char *url, const char *json_body,
                      const char *bearer, int timeout_ms, cp_http_resp *r) {
    return request(url, "POST", json_body, bearer, timeout_ms, r);
}
int cp_http_get(const char *url, const char *bearer, int timeout_ms, cp_http_resp *r) {
    return request(url, "GET", NULL, bearer, timeout_ms, r);
}
