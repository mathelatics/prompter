/* track — ASR front-end for Word-Tracking mode (M2).
 * Backend strategy (cprompter-02-deps.md): whisper.cpp is vendored under
 * third_party/ and compiled with WHISPER=1. Without it we still build two
 * testable backends:
 *   - "mock":  a scripted word stream fed at a fixed cadence (unit tests, CI)
 *   - "file":  reads a .wav via audio/wav.c, runs VAD-gated chunks through a
 *              pluggable recognize() fn (so real ASR drops in later)
 * The engine consumes cp_track_poll() -> array of newly recognized words.
 */
#ifndef CP_TRACK_H
#define CP_TRACK_H

#include <stddef.h>
#include "../audio/audio.h"

typedef struct cp_track cp_track;

typedef enum {
    CP_TRACK_NONE = 0,   /* disabled                                        */
    CP_TRACK_WHISPER,    /* whisper.cpp tiny/base int8 (WHISPER=1 builds)   */
    CP_TRACK_MOCK        /* synthetic stream for tests/demo                 */
} cp_track_backend;

typedef struct {
    cp_track_backend backend;
    const char      *model_path;   /* ggml-*.bin for whisper; NULL=default  */
    const char      *lang;         /* "en", "auto", ...                     */
    float            chunk_sec;    /* inference hop, default 1.0            */
    /* mock backend: */
    const char     **mock_words;   /* NULL-terminated array                 */
    double           mock_wps;     /* words per second fed to poller        */
} cp_track_params;

cp_track *cp_track_open(const cp_track_params *p, char *err, size_t errn);
void      cp_track_close(cp_track *t);

/* Attach live mic PCM (called from engine's audio callback). Internally
 * buffers ~chunk_sec of voiced audio and kicks recognition on a worker
 * thread; results queue up for cp_track_poll(). Safe to call often. */
void  cp_track_push_pcm(cp_track *t, const float *pcm, size_t n);

/* Drain newly completed recognitions. Returns count (<=max), each entry a
 * NUL-terminated recognized word. Words arrive in order. */
size_t cp_track_poll(cp_track *t, const char **out, size_t max);

/* reset internal audio buffer + sequence (e.g. after manual jump) */
void  cp_track_flush(cp_track *t);

const char *cp_track_backend_name(cp_track_backend b);

#endif /* CP_TRACK_H */
