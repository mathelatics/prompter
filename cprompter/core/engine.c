/* engine.c — thread-safe glue: audio cb -> vad/wpm -> pacer; UI reads via
 * mutex-protected getters. Pacer is ticked by the UI thread (owner of clock);
 * voice data flows in asynchronously. */
#define _POSIX_C_SOURCE 200809L   /* CLOCK_MONOTONIC under -std=c11 */
#include "engine.h"
#include "../audio/audio.h"
#include "../audio/vad.h"
#include "../audio/wpm.h"
#include "../audio/align.h"

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct cp_engine {
    cp_script     *s;              /* public: renderers may read          */
    cp_pacer      *p;              /* owned pointer, shared with UI     */
    pthread_mutex_t mu;
    cp_audio      *aud;
    cp_vad        *vad;
    cp_wpm        *wpm;
    cp_track      *track;          /* M2 ASR front-end (may be NULL)     */
    cp_align      *align;          /* M2 word tracker (may be NULL)      */
    int            audio_live;
    float          sensitivity;
};

double cp_engine_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec / 1e9;
}

static void on_pcm(const float *pcm, size_t n, void *user) {
    cp_engine *e = user;
    pthread_mutex_lock(&e->mu);
    int sp = cp_vad_push(e->vad, pcm, n);
    cp_wpm_push(e->wpm, pcm, n, sp);
    cp_pacer_feed_vad(e->p, sp, cp_wpm_estimate(e->wpm));
    if (e->track) cp_track_push_pcm(e->track, pcm, n);  /* M2: feed ASR too */
    pthread_mutex_unlock(&e->mu);
}

cp_engine *cp_engine_new(cp_script *s, cp_pacer *p,
                         const cp_engine_params *ep, char *err, size_t errn) {
    if (err && errn) err[0] = 0;
    cp_engine *e = calloc(1, sizeof *e);
    pthread_mutex_init(&e->mu, NULL);
    e->s = s; e->p = p;
    e->sensitivity = ep ? ep->sensitivity : 0.0f;

    e->vad = cp_vad_new(CP_AUDIO_RATE);
    e->wpm = cp_wpm_new(CP_AUDIO_RATE);
    cp_vad_set_sensitivity(e->vad, e->sensitivity);

    /* M2: word-tracking front-end (mock stream needs no mic; whisper does) */
    if (ep && ep->track.backend != CP_TRACK_NONE) {
        char terr[200] = "";
        e->track = cp_track_open(&ep->track, terr, sizeof terr);
        if (!e->track) {
            if (err) snprintf(err, errn, "track: %s", terr);
            cp_vad_free(e->vad); cp_wpm_free(e->wpm);
            pthread_mutex_destroy(&e->mu); free(e);
            return NULL;
        }
        e->align = cp_align_new(s);
    }

    int need_mic = ep && (ep->enable_audio ||
               (e->track && ep->track.backend == CP_TRACK_WHISPER));
    if (need_mic) {
        cp_audio_params ap = { .dev = ep->dev, .frame_ms = 20,
                               .cb = on_pcm, .user = e };
        char aerr[160] = "";
        e->aud = cp_audio_open(&ap, aerr, sizeof aerr);
        if (!e->aud) {
            if (err) snprintf(err, errn, "audio: %s", aerr);
            cp_vad_free(e->vad); cp_wpm_free(e->wpm);
            pthread_mutex_destroy(&e->mu); free(e);
            return NULL;
        }
        e->audio_live = 1;
    }
    return e;
}

void cp_engine_free(cp_engine *e) {
    if (!e) return;
    cp_engine_stop(e);
    if (e->aud) cp_audio_close(e->aud);
    if (e->track) cp_track_close(e->track);
    if (e->align) cp_align_free(e->align);
    cp_vad_free(e->vad); cp_wpm_free(e->wpm);
    pthread_mutex_destroy(&e->mu);
    free(e);
}

int cp_engine_start(cp_engine *e) {
    if (!e || !e->audio_live) return 0;
    return cp_audio_start(e->aud);
}

void cp_engine_stop(cp_engine *e) {
    if (e && e->audio_live) cp_audio_stop(e->aud);
}

const cp_pacer *cp_engine_pacer(const cp_engine *e) { return e->p; }

const cp_script *cp_engine_script(const cp_engine *e) { return e->s; }

int cp_engine_speaking(const cp_engine *e) {
    cp_engine *m = (cp_engine *)e;
    pthread_mutex_lock(&m->mu);
    int v = cp_vad_speaking(m->vad);
    pthread_mutex_unlock(&m->mu);
    return v;
}

float cp_engine_level(const cp_engine *e) {
    cp_engine *m = (cp_engine *)e;
    pthread_mutex_lock(&m->mu);
    float v = m->audio_live ? cp_vad_level(m->vad) : 0.0f;
    pthread_mutex_unlock(&m->mu);
    return v;
}

float cp_engine_wpm(const cp_engine *e) {
    cp_engine *m = (cp_engine *)e;
    pthread_mutex_lock(&m->mu);
    float v = cp_wpm_estimate(m->wpm);
    pthread_mutex_unlock(&m->mu);
    return v;
}

void cp_engine_set_speed(cp_engine *e, double wps) {
    pthread_mutex_lock(&e->mu);
    cp_pacer_set_speed(e->p, wps);
    pthread_mutex_unlock(&e->mu);
}

void cp_engine_toggle_pause(cp_engine *e) {
    pthread_mutex_lock(&e->mu);
    cp_pacer_set_paused(e->p, !e->p->paused);
    pthread_mutex_unlock(&e->mu);
}

void cp_engine_scroll(cp_engine *e, double dw) {
    pthread_mutex_lock(&e->mu);
    cp_pacer_scroll_by(e->p, dw);
    pthread_mutex_unlock(&e->mu);
}

void cp_engine_jump(cp_engine *e, size_t wi, int allow_back) {
    pthread_mutex_lock(&e->mu);
    cp_pacer_jump(e->p, (double)wi, allow_back);
    pthread_mutex_unlock(&e->mu);
}

void cp_engine_cycle_mode(cp_engine *e) {
    pthread_mutex_lock(&e->mu);
    int has_track = e->track && e->align;
    int next = (e->s->default_mode + 1) % 3;
    if (next == CP_MODE_TRACK && !has_track) next = CP_MODE_CLASSIC;
    e->s->default_mode = next;
    pthread_mutex_unlock(&e->mu);
}

/* M2: drain ASR words -> aligner -> pacer. UI-thread, once per frame. */
void cp_engine_update_track(cp_engine *e) {
    if (!e || !e->track || !e->align) return;
    const char *words[64];
    size_t n;
    pthread_mutex_lock(&e->mu);
    while ((n = cp_track_poll(e->track, words,
                              sizeof words / sizeof *words)) > 0) {
        for (size_t i = 0; i < n; i++)
            cp_pacer_feed_align(e->p, (double)cp_align_feed(e->align, words[i]).script_idx);
        if (n < sizeof words / sizeof *words) break;   /* queue drained */
    }
    pthread_mutex_unlock(&e->mu);
}

void cp_engine_resync(cp_engine *e) {
    if (!e) return;
    pthread_mutex_lock(&e->mu);
    cp_pacer_resync(e->p);
    if (e->align) cp_align_reset(e->align, cp_pacer_word_index(e->p));
    if (e->track) cp_track_flush(e->track);
    pthread_mutex_unlock(&e->mu);
}

int cp_engine_drift(const cp_engine *e) {
    if (!e || !e->align) return 0;
    pthread_mutex_lock((pthread_mutex_t *)&e->mu);
    int d = cp_align_drift(e->align);
    pthread_mutex_unlock((pthread_mutex_t *)&e->mu);
    return d;
}

const char *cp_engine_last_heard(const cp_engine *e) {
    if (!e || !e->align) return "";
    return cp_align_last_heard(e->align);
}

void cp_engine_vad_recalibrate(cp_engine *e) {
    pthread_mutex_lock(&e->mu);
    cp_vad_recalibrate(e->vad);
    cp_wpm_reset(e->wpm);
    pthread_mutex_unlock(&e->mu);
}

void cp_engine_sensitivity(cp_engine *e, float delta) {
    pthread_mutex_lock(&e->mu);
    e->sensitivity += delta;
    if (e->sensitivity > 2) e->sensitivity = 2;
    if (e->sensitivity < -2) e->sensitivity = -2;
    cp_vad_set_sensitivity(e->vad, e->sensitivity);
    pthread_mutex_unlock(&e->mu);
}

/* ---- M3: remote/director plumbing -------------------------------------- */

void cp_engine_snapshot(cp_engine *e, cp_engine_state *out) {
    if (!e || !out) return;
    memset(out, 0, sizeof *out);
    pthread_mutex_lock(&e->mu);
    const cp_pacer *p = e->p;
    out->pos           = p->pos;
    out->max_delivered = p->max_delivered;
    out->speed_wps     = p->speed_wps;
    out->nw            = e->s->nw;
    out->word_index    = (int)cp_pacer_word_index(p);
    out->page_index    = cp_page_of(e->s, (size_t)out->word_index);
    out->paused        = p->paused;
    out->finished      = cp_pacer_finished(p);
    out->mode          = (int)e->s->default_mode;
    out->speaking      = e->audio_live ? cp_vad_speaking(e->vad) : 0;
    out->level         = e->audio_live ? cp_vad_level(e->vad) : 0.0f;
    out->wpm           = cp_wpm_estimate(e->wpm);
    pthread_mutex_unlock(&e->mu);
}

int cp_engine_is_paused(const cp_engine *e) {
    cp_engine *m = (cp_engine *)e;
    pthread_mutex_lock(&m->mu);
    int v = m->p->paused;
    pthread_mutex_unlock(&m->mu);
    return v;
}

size_t cp_engine_nwords(const cp_engine *e) {
    cp_engine *m = (cp_engine *)e;
    pthread_mutex_lock(&m->mu);
    size_t v = m->s->nw;
    pthread_mutex_unlock(&m->mu);
    return v;
}

int cp_engine_apply_edit(cp_engine *e, const char *new_text,
                         char *err, size_t errn) {
    pthread_mutex_lock(&e->mu);
    int rc = cp_pacer_apply_edit(e->p, new_text, err, errn);
    pthread_mutex_unlock(&e->mu);
    return rc;
}

void cp_engine_after_edit(cp_engine *e) {
    if (!e) return;
    pthread_mutex_lock(&e->mu);
    if (e->align) cp_align_reset(e->align, cp_pacer_word_index(e->p));
    if (e->track) cp_track_flush(e->track);
    pthread_mutex_unlock(&e->mu);
}
