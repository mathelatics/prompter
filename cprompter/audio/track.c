/* track.c — backends: mock (always built) + whisper.cpp (WHISPER=1).
 * Threading model: push_pcm runs on the capture thread and only appends to a
 * ring buffer under mutex; a worker thread pops chunk_sec slices, runs ASR,
 * splits the transcript into words and pushes them to a result queue that
 * cp_track_poll() drains from the UI thread. All locks are try/avoid where
 * possible so a slow ASR never blocks audio capture. */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE               /* usleep() under glibc strict c11 */
#include "track.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef CP_HAVE_WHISPER
#include "whisper.h"
#endif

#define TR_RING_CAP   (CP_AUDIO_RATE * 30)   /* 30 s of float pcm          */
#define TR_WORD_Q     512                    /* queued recognized words    */
#define TR_WLEN       48

struct cp_track {
    cp_track_params p;
    float  chunk_sec;

    /* ring of pending pcm */
    float  ring[TR_RING_CAP];
    size_t rw;                 /* write pos (mod cap)                      */
    size_t fill;               /* valid samples                            */
    int    voiced_run;         /* consecutive voiced chunks heuristic      */
    int    silence_run;

    pthread_mutex_t mu;
    pthread_t  worker;
    int        quit, running;

    /* word result queue (producer: worker; consumer: poll) */
    char   words[TR_WORD_Q][TR_WLEN];
    size_t qh, qt, qn;

    /* mock backend state */
    size_t mock_i;
    double mock_next;          /* monotonic time of next word emission     */

#ifdef CP_HAVE_WHISPER
    struct whisper_context *wctx;
#endif
};

const char *cp_track_backend_name(cp_track_backend b) {
    switch (b) {
    case CP_TRACK_WHISPER: return "whisper";
    case CP_TRACK_MOCK:    return "mock";
    default:               return "none";
    }
}

static double mono_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void q_push(cp_track *t, const char *w) {
    if (t->qn >= TR_WORD_Q) return;           /* drop oldest? keep newest flow */
    size_t i = (t->qh + t->qn) % TR_WORD_Q;
    snprintf(t->words[i], TR_WLEN, "%s", w);
    t->qn++;
}

#ifdef CP_HAVE_WHISPER
/* split transcript on spaces/newlines, strip surrounding whitespace */
static void q_push_transcript(cp_track *t, const char *text) {
    const char *p = text;
    while (*p) {
        while (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r') p++;
        const char *s = p;
        while (*p && *p != ' ' && *p != '\n' && *p != '\t' && *p != '\r') p++;
        if (p > s) {
            char buf[TR_WLEN];
            size_t n = (size_t)(p - s);
            if (n >= TR_WLEN) n = TR_WLEN - 1;
            memcpy(buf, s, n); buf[n] = 0;
            q_push(t, buf);
        }
    }
}

/* run whisper full on `pcm` (mono float @16k). Appends tokens to queue. */
static void whisper_run_chunk(cp_track *t, const float *pcm, size_t n) {
    if (!t->wctx || n < (size_t)(CP_AUDIO_RATE * 0.3)) return;
    whisper_full_params wp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wp.print_progress = false; wp.print_special = false;
    wp.print_realtime = false; wp.print_timestamps = false;
    wp.language = t->p.lang ? t->p.lang : "en";
    wp.no_timestamps = true;
    wp.n_threads = 4;
    if (whisper_full(t->wctx, wp, pcm, (int)n) != 0) return;
    int nseg = whisper_full_n_segments(t->wctx);
    for (int i = 0; i < nseg; i++)
        q_push_transcript(t, whisper_full_get_segment_text(t->wctx, i));
}
#endif

static void *worker_main(void *arg) {
    cp_track *t = arg;
    float local[CP_AUDIO_RATE * 4];   /* up to 4 s slices */
    while (!t->quit) {
        size_t want = (size_t)(t->chunk_sec * CP_AUDIO_RATE);
        size_t got = 0;
        pthread_mutex_lock(&t->mu);
        got = t->fill < want ? t->fill : want;
        if (got) {
            size_t start = (t->rw - t->fill + TR_RING_CAP) % TR_RING_CAP;
            for (size_t i = 0; i < got; i++)
                local[i] = t->ring[(start + i) % TR_RING_CAP];
            t->fill -= got;
        }
        pthread_mutex_unlock(&t->mu);
        if (!got) { usleep(20 * 1000); continue; }
#ifdef CP_HAVE_WHISPER
        whisper_run_chunk(t, local, got);
#else
        (void)local; (void)got;
#endif
    }
    return NULL;
}

cp_track *cp_track_open(const cp_track_params *pp, char *err, size_t errn) {
    if (err && errn) err[0] = 0;
    if (!pp || pp->backend == CP_TRACK_NONE) return NULL;
    cp_track *t = calloc(1, sizeof *t);
    t->p = *pp;
    t->chunk_sec = pp->chunk_sec > 0.2f ? pp->chunk_sec : 1.0f;
    pthread_mutex_init(&t->mu, NULL);

    if (pp->backend == CP_TRACK_MOCK) {
        t->mock_next = mono_now();
        return t;
    }

#ifdef CP_HAVE_WHISPER
    struct whisper_context_params wc = whisper_context_default_params();
    t->wctx = whisper_init_from_file_with_params(pp->model_path
              ? pp->model_path : "models/ggml-base.en.bin", wc);
    if (!t->wctx) {
        if (err) snprintf(err, errn, "whisper: cannot load model");
        free(t); return NULL;
    }
    t->running = 1;
    pthread_create(&t->worker, NULL, worker_main, t);
    return t;
#else
    (void)worker_main;
    if (err) snprintf(err, errn,
        "built without whisper (make WHISPER=1, see third_party/whisper.cpp)");
    free(t);
    return NULL;
#endif
}

void cp_track_close(cp_track *t) {
    if (!t) return;
    if (t->running) {
        t->quit = 1;
        pthread_join(t->worker, NULL);
    }
#ifdef CP_HAVE_WHISPER
    if (t->wctx) whisper_free(t->wctx);
#endif
    pthread_mutex_destroy(&t->mu);
    free(t);
}

void cp_track_push_pcm(cp_track *t, const float *pcm, size_t n) {
    if (!t || t->p.backend == CP_TRACK_MOCK) return;   /* mock ignores pcm */
    pthread_mutex_lock(&t->mu);
    for (size_t i = 0; i < n; i++) {
        t->ring[t->rw] = pcm[i];
        t->rw = (t->rw + 1) % TR_RING_CAP;
        if (t->fill < TR_RING_CAP) t->fill++;
        else t->rw = (t->rw + 1) % TR_RING_CAP; /* overwrite oldest: advance read */
    }
    /* trim fill to cap implicitly: read pos = rw - fill */
    if (t->fill > TR_RING_CAP) t->fill = TR_RING_CAP;
    pthread_mutex_unlock(&t->mu);
}

void cp_track_flush(cp_track *t) {
    if (!t) return;
    pthread_mutex_lock(&t->mu);
    t->fill = 0; t->qh = t->qn = 0;
    t->mock_i = 0; t->mock_next = mono_now();
    pthread_mutex_unlock(&t->mu);
}

size_t cp_track_poll(cp_track *t, const char **out, size_t max) {
    if (!t || !max) return 0;
    size_t got = 0;

    if (t->p.backend == CP_TRACK_MOCK) {
        /* emit words at mock_wps based on wall clock. The UI thread polls
         * every frame (~60Hz), so we must lock around the cursor state —
         * flush() also takes this mutex. */
        pthread_mutex_lock(&t->mu);
        double now = mono_now();
        double wps = t->p.mock_wps > 0 ? t->p.mock_wps : 2.5;
        while (t->p.mock_words && t->p.mock_words[t->mock_i] && got < max) {
            if (now < t->mock_next) break;
            out[got++] = t->p.mock_words[t->mock_i++];
            t->mock_next += 1.0 / wps;
        }
        pthread_mutex_unlock(&t->mu);
        return got;
    }

    pthread_mutex_lock(&t->mu);
    while (t->qn && got < max) {
        out[got++] = t->words[t->qh];
        t->qh = (t->qh + 1) % TR_WORD_Q;
        t->qn--;
    }
    pthread_mutex_unlock(&t->mu);
    return got;
}
