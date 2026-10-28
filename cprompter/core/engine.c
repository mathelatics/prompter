/* engine.c — thread-safe glue: audio cb -> vad/wpm -> pacer; UI reads via
 * mutex-protected getters. Pacer is ticked by the UI thread (owner of clock);
 * voice data flows in asynchronously. */
#include "engine.h"
#include "../audio/audio.h"
#include "../audio/vad.h"
#include "../audio/wpm.h"

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

    if (ep && ep->enable_audio) {
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
    if (wps >= 0.5 && wps <= 8.0) e->p->speed_wps = wps;
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
    e->s->default_mode = (e->s->default_mode + 1) % 3;
    if (e->s->default_mode == CP_MODE_TRACK)   /* M2 not wired yet: skip */
        e->s->default_mode = CP_MODE_CLASSIC;
    pthread_mutex_unlock(&e->mu);
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
