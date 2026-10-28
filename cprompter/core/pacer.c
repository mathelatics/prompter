/* pacer.c — guidance-mode state machine. See plan §2: three modes feed one
 * fractional word position. Manual scroll pauses the clock and resumes from
 * the new position (Textream behavior). max_delivered is monotone -> drives
 * director read-lock (§4.5) and remote progress. */
#include "pacer.h"

#include <math.h>
#include <string.h>

static cp_mode active_mode(const cp_pacer *p) {
    size_t i = cp_pacer_word_index(p);
    int pg = cp_page_of(p->s, i);
    if (pg >= 0 && p->s->pages[pg].mode != (cp_mode)-1)
        return p->s->pages[pg].mode;
    return p->s->default_mode;
}

static double active_speed(const cp_pacer *p) {
    size_t i = cp_pacer_word_index(p);
    int pg = cp_page_of(p->s, i);
    if (pg >= 0 && p->s->pages[pg].scroll_speed > 0)
        return p->s->pages[pg].scroll_speed;
    return p->speed_wps > 0 ? p->speed_wps : p->s->default_speed;
}

static void clamp(cp_pacer *p) {
    double hi = (double)p->s->nw;
    if (p->pos < 0) p->pos = 0;
    if (p->pos > hi) p->pos = hi;
    if (p->pos > p->max_delivered) p->max_delivered = p->pos;
}

void cp_pacer_init(cp_pacer *p, const cp_script *s, double start_speed_wps) {
    memset(p, 0, sizeof *p);
    p->s = s;
    p->speed_wps = start_speed_wps > 0 ? start_speed_wps : s->default_speed;
    p->measured_wpm = 0;
    p->track_target = 0;
}

void cp_pacer_tick(cp_pacer *p, double now_sec, double dt) {
    if (dt <= 0) return;
    if (p->paused) { p->t_last = now_sec; return; }

    switch (active_mode(p)) {
    case CP_MODE_CLASSIC:
        p->pos += active_speed(p) * dt;
        break;
    case CP_MODE_VOICE:
        /* Textream: move only while speaking; rate = max(min speed, observed) */
        if (p->vad_active) {
            double sp = active_speed(p);
            double obs = p->measured_wpm / 60.0;
            p->pos += (obs > sp ? obs : sp) * dt;
        }
        break;
    case CP_MODE_TRACK:
        /* ease toward ASR-aligned target (critically-damped-ish lerp).
         * Never jump backward automatically (plan §4.1). */
        if (p->track_target > p->pos) {
            double gap = p->track_target - p->pos;
            double follow = gap * (1.0 - exp(-3.0 * dt));   /* ~0.95/s ease */
            if (follow > active_speed(p) * 2.5 * dt)        /* cap catch-up */
                follow = active_speed(p) * 2.5 * dt;
            p->pos += follow;
        }
        p->drifted_by = (int)(p->track_target - p->pos);
        break;
    }
    clamp(p);
    p->t_last = now_sec;
}

void cp_pacer_scroll_by(cp_pacer *p, double delta_words) {
    /* mouse-wheel override: repositions and keeps running from there */
    p->pos += delta_words;
    clamp(p);
}

void cp_pacer_jump(cp_pacer *p, double word_idx, int allow_back) {
    /* explicit jumps (tap-to-jump, keys): backward allowed only when asked */
    if (!allow_back && word_idx < p->pos) return;
    p->pos = word_idx;
    if (active_mode(p) == CP_MODE_TRACK) p->track_target = word_idx;
    clamp(p);
}

void cp_pacer_set_paused(cp_pacer *p, int paused) { p->paused = paused; }

void cp_pacer_feed_vad(cp_pacer *p, int active, double inst_wpm) {
    p->vad_active = active;
    if (active && inst_wpm > 0)
        p->measured_wpm = p->measured_wpm <= 0
                        ? inst_wpm
                        : p->measured_wpm * 0.8 + inst_wpm * 0.2;  /* EMA */
}

void cp_pacer_feed_align(cp_pacer *p, double aligned_pos) {
    if (aligned_pos > p->track_target)      /* forward-only for auto track */
        p->track_target = aligned_pos;
}

size_t cp_pacer_word_index(const cp_pacer *p) {
    long i = (long)floor(p->pos);
    if (i < 0) i = 0;
    if ((size_t)i >= p->s->nw) i = (long)(p->s->nw ? p->s->nw - 1 : 0);
    return (size_t)i;
}

int cp_pacer_finished(const cp_pacer *p) {
    return p->s->nw > 0 && p->pos >= (double)p->s->nw - 0.001;
}

double cp_pacer_remaining_seconds(const cp_pacer *p) {
    double rem = (double)p->s->nw - p->pos;
    double sp = active_speed(p);
    if (sp <= 0) sp = 1e-9;
    return rem > 0 ? rem / sp : 0;
}
