/* pacer — the single source of truth for "where are we in the script".
 * Three guidance modes feed one position; everything downstream (renderer,
 * remote sync at 10Hz, director read-lock, AI coach) consumes pacer state. */
#ifndef CP_PACER_H
#define CP_PACER_H

#include "script.h"

typedef struct cp_pacer {
    const cp_script *s;
    /* continuous fractional word position (like Textream's smooth scroll) */
    double  pos;          /* current word index, fractional              */
    double  max_delivered;/* monotone high-water mark -> read-lock       */
    int     paused;       /* spacebar / manual-scroll pause               */
    double  speed_wps;    /* live words-per-second (mutable mid-show)     */
    double  t_last;       /* seconds timestamp of last update            */
    /* voice mode state (fed by VAD in M1) */
    int     vad_active;   /* 1 = speech detected now                     */
    double  measured_wpm; /* smoothed observed speaking rate             */
    /* tracking mode state (fed by align.c in M2) */
    double  track_target; /* aligned cursor from ASR, we ease toward it  */
    int     drifted_by;   /* words behind/ahead vs plan, for coach       */
} cp_pacer;

void   cp_pacer_init(cp_pacer *p, const cp_script *s, double start_speed_wps);

/* Advance clock by dt seconds using the active mode. Call once per frame. */
void   cp_pacer_tick(cp_pacer *p, double now_sec, double dt);

/* External nudges */
void   cp_pacer_scroll_by(cp_pacer *p, double delta_words); /* mouse wheel */
void   cp_pacer_jump(cp_pacer *p, double word_idx, int allow_back);
void   cp_pacer_set_paused(cp_pacer *p, int paused);
void   cp_pacer_feed_vad(cp_pacer *p, int active, double inst_wpm);
void   cp_pacer_feed_align(cp_pacer *p, double aligned_pos);
void   cp_pacer_resync(cp_pacer *p);            /* snap pos to track_target */

/* Queries */
size_t cp_pacer_word_index(const cp_pacer *p);        /* floor(pos) clamped */
int    cp_pacer_finished(const cp_pacer *p);
double cp_pacer_remaining_seconds(const cp_pacer *p);

/* M3 director read-lock (see pacer.c comment): chars of s->raw covering the
 * first `words` delivered words. 0 before speech starts, len when done. */
size_t cp_script_chars_for_words(const cp_script *s, double words);
/* Apply a full-script text replacement from Director Mode. Returns 0 and
 * re-tokenizes the script in place; returns -1 (and fills err) if the edit
 * touches already-delivered words (read-lock -> HTTP 409 upstream). */
int    cp_pacer_apply_edit(cp_pacer *p, const char *new_text,
                           char *err, size_t errn);

#endif /* CP_PACER_H */
