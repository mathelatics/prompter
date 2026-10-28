/* wpm — syllabic speaking-rate estimator for Voice mode pacing + AI coach.
 * Classic speech-science trick: count syllable peaks in the envelope
 * (t~150 ms), divide by elapsed time, scale by words-per-syllable (~1.4 for
 * English). No ASR needed; robust to pauses; feeds cp_pacer_feed_vad(). */
#ifndef CP_WPM_H
#define CP_WPM_H

#include <stddef.h>

typedef struct cp_wpm cp_wpm;

cp_wpm *cp_wpm_new(int sample_rate);
void    cp_wpm_free(cp_wpm *w);

/* push a frame of (already VAD-gated or raw) float pcm @ rate */
void  cp_wpm_push(cp_wpm *w, const float *pcm, size_t n, int voiced);

/* smoothed estimate over ~last 6 s. inst = current utterance only. */
float cp_wpm_estimate(const cp_wpm *w);
float cp_wpm_inst(const cp_wpm *w);

/* syllable events for waveform UI ticks */
int   cp_wpm_syllables_total(const cp_wpm *w);
void  cp_wpm_reset(cp_wpm *w);

#endif
