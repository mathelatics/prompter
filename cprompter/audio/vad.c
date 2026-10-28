/* vad.c — energy + periodicity VAD with adaptive noise floor & hangover.
 * Tuning targets (from Textream UX): freeze within ~150 ms of stop-talking,
 * never miss inter-word pauses < 300 ms, tolerate steady HVAC hiss. */
#include "vad.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define VAD_WIN   512     /* analysis window (32 ms @16k)            */
#define VAD_HOP   256     /* hop (16 ms)                              */
#define HANGOVER_MS 300

struct cp_vad {
    int   rate;
    float win[VAD_WIN];
    size_t have;              /* samples buffered toward next frame   */
    /* noise-floor tracker (min-statistics style) */
    float nf;                 /* noise floor RMS                      */
    float peak;               /* running speech-level estimate        */
    int   nf_frames;
    /* decision state */
    int   speaking;
    int   hang_ms;            /* remaining hangover                   */
    float level;              /* smoothed RMS for UI                  */
    float last_rms;
    float sens;               /* -2..+2                               */
    int   primed;             /* skip first few frames                */
};

cp_vad *cp_vad_new(int sample_rate) {
    cp_vad *v = calloc(1, sizeof *v);
    v->rate = sample_rate > 0 ? sample_rate : 16000;
    v->nf = 1e-4f;           /* start assuming a quiet room */
    v->peak = 0.02f;
    return v;
}

void cp_vad_free(cp_vad *v) { free(v); }

/* periodicity score: best normalized autocorrelation in the voicing band
 * (70..350 Hz). Voiced speech -> high (>0.4); hiss/white noise -> low. */
static float periodicity(const float *x, int rate) {
    enum { MAXLAG = 230 };
    double e0 = 0.0;
    for (int i = 0; i < VAD_WIN; i++) e0 += (double)x[i] * x[i];
    if (e0 < 1e-9) return 0.0f;
    int lmin = rate / 350, lmax = rate / 70;
    if (lmax > MAXLAG) lmax = MAXLAG;
    if (lmin < 20) lmin = 20;
    float best = 0.0f;
    for (int lag = lmin; lag <= lmax; lag++) {
        double r = 0.0, e1 = 0.0;
        for (int i = 0; i + lag < VAD_WIN; i++) {
            r  += (double)x[i] * x[i + lag];
            e1 += (double)x[i + lag] * x[i + lag];
        }
        double norm = r / sqrt(e0 * e1 + 1e-12);
        if (norm > best) best = (float)norm;
    }
    return best;
}

static void process_frame(cp_vad *v, const float *x) {
    float rms = 0.0f;
    for (int i = 0; i < VAD_WIN; i++) rms += x[i] * x[i];
    rms = sqrtf(rms / VAD_WIN);
    v->last_rms = rms;
    v->level = v->level * 0.85f + rms * 0.15f;

    /* noise floor: track minimum with slow rise on sustained lows */
    if (!v->primed) {
        /* first 20 frames: pure learning of the room */
        v->nf = v->nf * 0.9f + rms * 0.1f;
        if (++v->nf_frames >= 20) v->primed = 1;
    } else {
        if (rms < v->nf) v->nf = v->nf * 0.99f + rms * 0.01f;      /* fast drop */
        else if (rms < v->nf * 1.6f) v->nf = v->nf * 0.9995f + rms * 0.0005f;
    }

    /* speech score: energy over noise floor, gated by periodicity so that
     * broadband hiss just above nf doesn't trigger */
    float snr = rms / (v->nf + 1e-9f);
    float per = periodicity(x, v->rate);
    float sens_adj = 1.0f + v->sens * 0.25f;         /* higher sens -> lower bar */
    float need = 2.5f / sens_adj;                    /* SNR threshold            */
    int voice = (snr > need) && (per > 0.35f);

    /* peak tracker for dynamic range display */
    if (voice && rms > v->peak) v->peak = v->peak * 0.7f + rms * 0.3f;

    int hop_ms = (int)(1000.0 * VAD_HOP / v->rate);
    if (voice) {
        v->speaking = 1;
        v->hang_ms = HANGOVER_MS;
    } else if (v->speaking) {
        v->hang_ms -= hop_ms;
        if (v->hang_ms <= 0) { v->speaking = 0; v->hang_ms = 0; }
    }
}

int cp_vad_push(cp_vad *v, const float *pcm, size_t n) {
    /* buffer arbitrary chunk sizes into VAD_WIN hops */
    for (size_t i = 0; i < n; i++) {
        v->win[v->have++] = pcm[i];
        if (v->have == VAD_HOP) {          /* slide: keep VAD_WIN-VAD_HOP old */
            memmove(v->win, v->win + VAD_HOP,
                    (VAD_WIN - VAD_HOP) * sizeof(float));
            v->have = VAD_WIN - VAD_HOP;
        }
        if (v->have == VAD_WIN) {
            process_frame(v, v->win);
            memmove(v->win, v->win + VAD_HOP,
                    (VAD_WIN - VAD_HOP) * sizeof(float));
            v->have = VAD_WIN - VAD_HOP;
        }
    }
    return v->speaking;
}

int   cp_vad_speaking(const cp_vad *v) { return v->speaking; }
float cp_vad_level(const cp_vad *v)    { return v->level; }
float cp_vad_rms(const cp_vad *v)      { return v->last_rms; }
float cp_vad_thresh(const cp_vad *v)   { return v->nf * 2.5f / (1.0f + v->sens * 0.25f); }

void cp_vad_set_sensitivity(cp_vad *v, float s) {
    if (s < -2) s = -2;
    if (s > 2)  s = 2;
    v->sens = s;
}

void cp_vad_recalibrate(cp_vad *v) {
    v->primed = 0; v->nf_frames = 0; v->speaking = 0; v->hang_ms = 0;
}
