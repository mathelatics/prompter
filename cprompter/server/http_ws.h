/* http_ws.h — tiny HTTP/1.1 + WebSocket(RFC6455) server over POSIX sockets.
 * Single accept thread, one handler callback per request; WS upgrade is
 * handled internally (Sec-WebSocket-Accept handshake, text frames, ping/pong,
 * close). Built for cPrompter M3: :7373 read-only viewer, :7575 director.
 * No external deps (civetweb not needed at this scale: localhost LAN, few
 * clients, 10Hz pushes). */
#ifndef CP_HTTP_WS_H
#define CP_HTTP_WS_H

#include <stddef.h>

typedef struct cp_httpd cp_httpd;

/* Request description passed to the handler. */
typedef struct {
    const char *method;              /* "GET", "POST", ...                   */
    const char *path;                /* "/api/state" (query stripped)        */
    const char *query;               /* raw query string or ""               */
    const char *body;                /* request body (NUL-terminated) or ""  */
    size_t      body_len;
    const char *content_type;        /* or ""                                */
    /* response to fill in: */
    int   status;                    /* default 200 if 0                     */
    char *ctype;  size_t ctype_cap;  /* e.g. "application/json"              */
    char *out;    size_t out_cap;    /* body buffer (non-WS responses)       */
    size_t out_len;                  /* set by handler                       */
    void *user;                      /* opaque from cp_httpd_serve           */
    int   want_ws;                   /* 1 = client requested WS upgrade      */
} cp_http_req;

/* WS connection handle: push text frames after the upgrade. */
typedef struct cp_ws cp_ws;
void cp_ws_send_text(cp_ws *w, const char *text);
int  cp_ws_is_open(cp_ws *w);

typedef void (*cp_http_handler)(cp_http_req *r);
typedef void (*cp_ws_open_cb)(cp_ws *w, cp_http_req *r, void *user);

cp_httpd *cp_httpd_start(int port, cp_http_handler h, void *user);
void      cp_httpd_stop(cp_httpd *d);

/* Broadcast helpers: the server tracks upgraded WS clients internally. */
size_t    cp_httpd_ws_count(cp_httpd *d);
void      cp_httpd_ws_broadcast(cp_httpd *d, const char *text);

/* Attach an open-cb called right after each successful upgrade. */
void      cp_httpd_on_ws_open(cp_httpd *d, cp_ws_open_cb cb);

#endif /* CP_HTTP_WS_H */
