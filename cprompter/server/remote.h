/* remote.h — M3: Remote Viewer (:7373) + Director Mode (:7575) for cPrompter.
 * Wraps two cp_httpd servers around a live cp_engine:
 *   :7373  GET /            -> read-only viewer page (word highlight, progress)
 *          GET /api/state    -> JSON snapshot (polling fallback)
 *          WS  /ws           -> 10 Hz state pushes {pos,maxDelivered,speed,
 *                                wordIndex,pageIndex,paused,speaking,finished,
 *                                level,wpm,mode}
 *   :7575  GET /            -> director editor page (write, push, read-lock)
 *          POST /api/script  -> apply edit; 409 if it touches delivered text
 *          POST /api/pause|speed|jump -> live controls
 *          WS  /ws           -> same state stream + script_changed events
 * QR codes (vendor/qrcodegen) are printed to the terminal on start and served
 * as SVG at /qr.svg so phones can scan straight from the desktop. */
#ifndef CP_REMOTE_H
#define CP_REMOTE_H

#include <stddef.h>
#include "../core/engine.h"

typedef struct cp_remote cp_remote;

cp_remote *cp_remote_start(cp_engine *eng, int viewer_port, int director_port);
void       cp_remote_stop(cp_remote *r);

/* Call from the UI thread ~every frame; throttles pushes to 10 Hz. */
void       cp_remote_tick(cp_remote *r);

/* Print both QR codes (viewer/director URLs) to stdout. */
void       cp_remote_print_qr(cp_remote *r);

/* Number of connected WS clients (for status line/tests). */
size_t     cp_remote_viewers(cp_remote *r);
size_t     cp_remote_directors(cp_remote *r);

#endif /* CP_REMOTE_H */
