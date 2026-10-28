/* vad — voice activity gate for Voice-Activated mode (plan §4.2).
 * Pure C, no deps: frame energy (RMS) + spectral-flatness heuristic with a
 * 300 ms hangover so mid-word breaths don't freeze the scroll. API is fed by
 * the audio callback; UI polls cp_vad_speaking()/cp_vad_level().
 * Later M2 can swap this for webrtc-vad without touching callers. */
#ifndef CP_VAD_H
#define CP_VAD_H

#include <stddef.h>

typedef struct cp_vad cp_vad;

cp_vad *cp_vad_new(int sample_rate);
void    cp_vad_free(cp_vad *v);

/* feed one analysis frame (mono float @ v's rate). Returns 1 if speech now. */
int   cp_vad_push(cp_vad *v, const float *pcm, size_t n);

int   cp_vad_speaking(const cp_vad *v);   /* last decision (hangover applied) */
float cp_vad_level(const cp_vad *v);      /* smoothed RMS 0..1 for waveform   */
float cp_vad_rms(const cp_vad *v);        /* raw recent RMS                   */
float cp_vad_thresh(const cp_vad *v);     /* current adaptive threshold       */

/* calibration: raise/lower sensitivity (-2..+2), and noise-floor relearn */
void  cp_vad_set_sensitivity(cp_vad *v, float s);
void  cp_vad_recalibrate(cp_vad *v);

#endif /* CP_VAD_H */
