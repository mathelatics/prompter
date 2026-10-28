/* audio.c — capture backends + DSP helpers. Zero build-deps: ALSA is loaded
 * at runtime via dlopen("libasound.so.2"); dev=="null" gives a deterministic
 * test signal (silence, or 1 kHz tone when CP_NULL_TONE=1) for CI.
 * Portability note: macOS CoreAudio / Windows WASAPI backends slot in here
 * later (plan §6 cross-build); the cb contract never changes. */
#define _GNU_SOURCE
#include "audio.h"

#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* DSP helpers                                                         */
/* ------------------------------------------------------------------ */

size_t cp_resample_i16_to16k(const int16_t *src, size_t n_in, int src_rate,
                             float *dst, size_t dst_cap) {
    if (src_rate <= 0 || !n_in) return 0;
    /* integer ratio averaging: out[j] = mean of samples in window */
    double step = (double)src_rate / CP_AUDIO_RATE;   /* src per dst sample */
    size_t produced = 0;
    double pos = 0.0;
    while (1) {
        size_t i0 = (size_t)floor(pos);
        size_t i1 = (size_t)floor(pos + step);
        if (i0 >= n_in) break;
        if (i1 > n_in) i1 = n_in;
        if (i1 <= i0) i1 = i0 + 1;
        double acc = 0.0;
        for (size_t i = i0; i < i1; i++) acc += (double)src[i];
        float v = (float)(acc / (double)(i1 - i0) / 32768.0);
        if (produced >= dst_cap) break;
        dst[produced++] = v;
        pos += step;
    }
    return produced;
}

float cp_rms(const float *pcm, size_t n) {
    if (!n) return 0.0f;
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) acc += (double)pcm[i] * pcm[i];
    return (float)sqrt(acc / (double)n);
}

void cp_hp1_reset(cp_hp1 *h) { h->x1 = 0.0f; h->y1 = 0.0f; }

void cp_hp1_run(cp_hp1 *h, float *pcm, size_t n) {
    /* y[n] = a*(y[n-1] + x[n] - x[n-1]), a ~ 0.995 -> fc ~= 8Hz @16k */
    const float a = 0.995f;
    for (size_t i = 0; i < n; i++) {
        float x = pcm[i];
        float y = a * (h->y1 + x - h->x1);
        h->x1 = x; h->y1 = y;
        pcm[i] = y;
    }
}

/* ------------------------------------------------------------------ */
/* core struct                                                         */
/* ------------------------------------------------------------------ */

struct cp_audio {
    cp_audio_params p;
    pthread_t       th;
    volatile int    running, want_stop;
    volatile float  level;         /* 0..1, last frame RMS             */
    /* backend state */
    void           *handle;        /* snd_ctl_handle etc (ALSA)        */
    void           *pcm;           /* snd_pcm_t*                       */
    int             null_mode;
};

/* ------------------------------------------------------------------ */
/* ALSA backend via dlopen                                             */
/* ------------------------------------------------------------------ */

/* minimal type defs to avoid needing alsa headers at compile time
 * (values match the stable ALSA ABI: include/pcm.h, include/soundlib.h) */
typedef void *snd_pcm_t_ptr;
typedef long  snd_pcm_sframes_t;
typedef unsigned long snd_pcm_uframes_t;
typedef int   errcode_t;

typedef struct {
    int (*open)(snd_pcm_t_ptr *pcm, const char *name, int stream,
                int mode, const char *app_name);
    int (*set_params)(snd_pcm_t_ptr pcm, int access, int fmt,
                      unsigned channels, unsigned rate, unsigned period_dirs,
                      unsigned period_size, unsigned start_threshold,
                      unsigned stop_threshold, unsigned silence_threshold);
    snd_pcm_sframes_t (*readi)(snd_pcm_t_ptr pcm, void *buffer,
                               snd_pcm_uframes_t size);
    int (*close)(snd_pcm_t_ptr pcm);
    int (*drain)(snd_pcm_t_ptr pcm);
    const char *(*strerror)(errcode_t);
} alsa_api;

static alsa_api A;
static void *alsa_lib;

static int alsa_try_load(char *err, size_t errn) {
    if (alsa_lib) return 0;
    alsa_lib = dlopen("libasound.so.2", RTLD_NOW | RTLD_LOCAL);
    if (!alsa_lib) alsa_lib = dlopen("libasound.so", RTLD_NOW | RTLD_LOCAL);
    if (!alsa_lib) { snprintf(err, errn, "libasound not found"); return -1; }
    void *sym;
    #define GET(field, name) \
        do { sym = dlsym(alsa_lib, name); if (!sym) { \
            snprintf(err, errn, "missing symbol %s", name); \
            dlclose(alsa_lib); alsa_lib = NULL; return -1; } \
            *(void**)&A.field = sym; } while (0)
    GET(open,          "snd_pcm_open");
    GET(set_params,    "snd_pcm_set_params");
    GET(readi,         "snd_pcm_readi");
    GET(close,         "snd_pcm_close");
    GET(drain,         "snd_pcm_drain");
    GET(strerror,      "snd_strerror");
    #undef GET
    return 0;
}

/* constants we need from alsa (stable ABI values) */
enum { SND_STREAM_CAPTURE = 0 };
enum { SND_MODE_BLOCKING = 0 };
enum { SND_ACCESS_MMAP_INTERLEAVED = 4, SND_ACCESS_RW_INTERLEAVED = 3 };
enum { SND_FORMAT_S16_LE = 2 };

/* ------------------------------------------------------------------ */
/* null backend                                                        */
/* ------------------------------------------------------------------ */

static void *thread_null(void *arg) {
    cp_audio *a = arg;
    size_t fn = (size_t)CP_AUDIO_RATE * (size_t)(a->p.frame_ms ? a->p.frame_ms : 20) / 1000;
    float *buf = calloc(fn, sizeof *buf);
    const char *tone = getenv("CP_NULL_TONE");
    double ph = 0.0;
    while (!a->want_stop) {
        if (tone && *tone == '1') {
            for (size_t i = 0; i < fn; i++) {
                ph += 2.0 * M_PI * 1000.0 / CP_AUDIO_RATE;
                buf[i] = 0.3f * (float)sin(ph);
            }
        } else {
            memset(buf, 0, fn * sizeof *buf);
        }
        a->level = cp_rms(buf, fn);
        if (a->p.cb) a->p.cb(buf, fn, a->p.user);
        usleep((useconds_t)(a->p.frame_ms ? a->p.frame_ms : 20) * 1000);
    }
    free(buf);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* ALSA thread                                                         */
/* ------------------------------------------------------------------ */

static void *thread_alsa(void *arg) {
    cp_audio *a = arg;
    snd_pcm_t_ptr pcm = NULL;
    int rc = A.open(&pcm, a->p.dev ? a->p.dev : "default",
                    SND_STREAM_CAPTURE, SND_MODE_BLOCKING, "cprompter");
    if (rc < 0) { fprintf(stderr, "cprompter audio: snd_pcm_open: %s\n", A.strerror(rc)); a->running = 0; return NULL; }
    unsigned period = (unsigned)((long)CP_AUDIO_RATE * (a->p.frame_ms ? a->p.frame_ms : 20) / 1000);
    rc = A.set_params(pcm, SND_ACCESS_RW_INTERLEAVED, SND_FORMAT_S16_LE,
                      1u /*mono*/, (unsigned)CP_AUDIO_RATE,
                      0 /*period dirs default*/, period,
                      period * 2, period * 4, 0);
    if (rc < 0) { fprintf(stderr, "cprompter audio: set_params: %s\n", A.strerror(rc)); A.close(pcm); a->running = 0; return NULL; }

    int16_t *raw = malloc(period * sizeof *raw);
    while (!a->want_stop) {
        snd_pcm_sframes_t got = A.readi(pcm, raw, period);
        if (got < 0) { /* xrun-ish: keep going, ALSA recovered internally */
            usleep(10000); continue;
        }
        /* ALSA already gave us 16k mono floats via conversion? no: S16. */
        static __thread float fbuf[4096];
        size_t nf = (size_t)got;
        if (nf > 4096) nf = 4096;
        for (size_t i = 0; i < nf; i++) fbuf[i] = (float)raw[i] / 32768.0f;
        a->level = cp_rms(fbuf, nf);
        if (a->p.cb) a->p.cb(fbuf, nf, a->p.user);
    }
    free(raw);
    A.drain(pcm);
    A.close(pcm);
    a->running = 0;
    return NULL;
}

/* ------------------------------------------------------------------ */
/* public API                                                          */
/* ------------------------------------------------------------------ */

cp_audio *cp_audio_open(const cp_audio_params *p, char *err, size_t errn) {
    if (err && errn) err[0] = 0;
    if (!p || !p->cb) { if (err) snprintf(err, errn, "cb required"); return NULL; }
    cp_audio *a = calloc(1, sizeof *a);
    a->p = *p;
    if (a->p.frame_ms <= 0) a->p.frame_ms = 20;
    if (a->p.frame_ms > 100) a->p.frame_ms = 100;

    if (p->dev && !strcmp(p->dev, "null")) { a->null_mode = 1; return a; }

    char aerr[128] = "";
    if (alsa_try_load(aerr, sizeof aerr) != 0) {
        /* fall back to null so the show must go on */
        a->null_mode = 1;
        fprintf(stderr, "cprompter audio: %s -> using null backend\n", aerr);
    }
    return a;
}

int cp_audio_start(cp_audio *a) {
    if (!a || a->running) return -1;
    a->want_stop = 0; a->running = 1;
    if (pthread_create(&a->th, NULL, a->null_mode ? thread_null : thread_alsa, a) != 0)
        { a->running = 0; return -1; }
    return 0;
}

void cp_audio_stop(cp_audio *a) {
    if (!a || !a->running) return;
    a->want_stop = 1;
    pthread_join(a->th, NULL);
    a->running = 0;
}

void cp_audio_close(cp_audio *a) { if (a) free(a); }

float cp_audio_level(cp_audio *a) { return a ? a->level : 0.0f; }
