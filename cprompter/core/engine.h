/* engine — wires pacer + audio + vad + wpm into one runtime object and owns
 * the interactive loop. Used by main.c for classic/voice modes; M2 will add
 * cp_engine_feed_asr() for track mode without changing callers. */
#ifndef CP_ENGINE_H
#define CP_ENGINE_H

#include "../core/script.h"
#include "../core/pacer.h"
#include "../audio/track.h"

typedef struct cp_engine cp_engine;

typedef struct {
    const char *dev;         /* NULL = default; "null" = silent test   */
    int         enable_audio;/* 0 -> pure clock (classic), no mic      */
    float       sensitivity; /* VAD -2..+2                               */
    /* M2 word tracking: backend CP_TRACK_NONE disables. Mock stream is
     * used by tests/demo; whisper requires WHISPER=1 build. */
    cp_track_params track;
} cp_engine_params;

cp_engine *cp_engine_new(cp_script *s, cp_pacer *p,
                         const cp_engine_params *ep, char *err, size_t errn);
void       cp_engine_free(cp_engine *e);

int  cp_engine_start(cp_engine *e);
void cp_engine_stop(cp_engine *e);

/* shared-state getters for renderers (called from UI thread) */
const cp_pacer   *cp_engine_pacer(const cp_engine *e);
int               cp_engine_speaking(const cp_engine *e);
float             cp_engine_level(const cp_engine *e);      /* 0..1 waveform */
float             cp_engine_wpm(const cp_engine *e);
double            cp_engine_now(void);                       /* monotonic sec */
/* M2 trackers (safe with tracking disabled: return 0/NULL) */
int               cp_engine_drift(const cp_engine *e);      /* miss streak   */
const char       *cp_engine_last_heard(const cp_engine *e); /* ASR ticker    */

/* live controls */
void cp_engine_set_speed(cp_engine *e, double wps);
void cp_engine_toggle_pause(cp_engine *e);
void cp_engine_scroll(cp_engine *e, double dw);
void cp_engine_jump(cp_engine *e, size_t word_idx, int allow_back);
void cp_engine_cycle_mode(cp_engine *e);
void cp_engine_vad_recalibrate(cp_engine *e);
void cp_engine_sensitivity(cp_engine *e, float delta);
/* M2: pull recognized words -> aligner -> pacer. Call once per UI frame. */
void cp_engine_update_track(cp_engine *e);
/* M2: snap display position onto the tracker cursor (tap-to-resync) */
void cp_engine_resync(cp_engine *e);

/* ---- M3 remote/director plumbing (server/remote.c consumers) ---- */
/* one-stop snapshot for the 10 Hz push; any field may be NULL */
typedef struct {
    double pos, max_delivered, speed_wps;
    size_t nw;
    int    word_index, page_index, paused, speaking, finished;
    float  level;          /* mic 0..1 (0 without audio)        */
    float  wpm;            /* observed speaking rate (0 = none) */
    int    mode;           /* cp_mode                           */
} cp_engine_state;
void cp_engine_snapshot(cp_engine *e, cp_engine_state *out);

int   cp_engine_is_paused(const cp_engine *e);
size_t cp_engine_nwords(const cp_engine *e);
/* director edit (read-lock enforced inside pacer): 0 ok, -1 conflict + err */
int   cp_engine_apply_edit(cp_engine *e, const char *new_text,
                           char *err, size_t errn);
/* re-point derived trackers after an in-place script change */
void  cp_engine_after_edit(cp_engine *e);

#endif
