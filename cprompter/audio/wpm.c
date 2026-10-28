/* wpm.c — envelope-peak syllable counting -> words/min estimate. */
#include "wpm.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define WPM_HIST 600        /* 10 ms bins * 60 s window */

struct cp_wpm {
    int    rate;
    float  env;             /* smoothed envelope (attack/release)     */
    float  prev_env;
    int    in_peak;         /* currently above the local-max line     */
    float  peak_val;
    double t_syll_last;     /* seconds of last counted syllable       */
    double t_now;           /* clock advanced by push()               */
    double t_speech_start;  /* start of current utterance (voiced run)*/
    int    voiced_run;
    /* rolling history of syllable times for smoothed estimate */
    double syl[WPM_HIST];
    int    syl_head, syl_n;
    int    total;
    float  est, inst;
};

cp_wpm *cp_wpm_new(int sample_rate) {
    cp_wpm *w = calloc(1, sizeof *w);
    w->rate = sample_rate > 0 ? sample_rate : 16000;
    return w;
}

void cp_wpm_free(cp_wpm *w) { free(w); }

void cp_wpm_reset(cp_wpm *w) {
    w->env = w->prev_env = 0; w->in_peak = 0; w->peak_val = 0;
    w->syl_head = w->syl_n = 0; w->total = 0;
    w->est = w->inst = 0; w->t_speech_start = 0; w->voiced_run = 0;
}

static void note_syllable(cp_wpm *w, double t) {
    /* refractory: syllables are 120..300 ms apart; ignore closer than 90 ms */
    if (t - w->t_syll_last < 0.09) return;
    w->t_syll_last = t;
    w->syl[w->syl_head] = t;
    w->syl_head = (w->syl_head + 1) % WPM_HIST;
    if (w->syl_n < WPM_HIST) w->syl_n++;
    w->total++;

    /* smoothed: syllables/sec over min(6s, utterance) window */
    double cutoff = t - 6.0;
    int cnt = 0;
    for (int i = 0; i < w->syl_n; i++)
        if (w->syl[i] >= cutoff) cnt++;
    double span = t - (cnt ? w->syl[(w->syl_head - cnt + WPM_HIST) % WPM_HIST] : t);
    if (span < 1.0) span = 1.0;
    double sps = cnt / span;
    w->est = (float)(sps * 60.0 / 1.4);      /* ~1.4 syllables per word (en) */
    if (w->est > 400) w->est = 400;          /* sanity clamp                 */

    /* instantaneous: since utterance start */
    if (w->t_speech_start > 0 && t > w->t_speech_start + 0.5) {
        double u_span = t - w->t_speech_start;
        double usyl = 0;
        for (int i = 0; i < w->syl_n; i++)
            if (w->syl[i] >= w->t_speech_start) usyl++;
        w->inst = (float)(usyl / u_span * 60.0 / 1.4);
    }
}

void cp_wpm_push(cp_wpm *w, const float *pcm, size_t n, int voiced) {
    double dt = (double)n / w->rate;
    w->t_now += dt;

    if (voiced) {
        if (!w->voiced_run) { w->voiced_run = 1; w->t_speech_start = w->t_now; }
    } else {
        if (w->voiced_run) { w->voiced_run = 0; w->inst = 0; }
    }

    /* envelope with fast attack / slower release (~speech amplitude curve) */
    float e = 0.0f;
    for (size_t i = 0; i < n; i++) { float a = fabsf(pcm[i]); if (a > e) e = a; }
    float atk = 0.5f, rel = 0.08f;
    w->prev_env = w->env;
    w->env = (e > w->env) ? w->env + atk * (e - w->env)
                          : w->env + rel * (e - w->env);

    if (!voiced) return;

    /* peak-pick on envelope: rise then fall through midline */
    if (!w->in_peak && w->env > w->prev_env && w->env > 0.02f) {
        w->in_peak = 1; w->peak_val = w->env;
    } else if (w->in_peak) {
        if (w->env > w->peak_val) w->peak_val = w->env;
        else if (w->env < w->peak_val * 0.6f) {   /* clear fall -> count it */
            w->in_peak = 0;
            note_syllable(w, w->t_now);
        }
    }
}

float cp_wpm_estimate(const cp_wpm *w) { return w->est; }
float cp_wpm_inst(const cp_wpm *w)     { return w->inst; }
int   cp_wpm_syllables_total(const cp_wpm *w) { return w->total; }
