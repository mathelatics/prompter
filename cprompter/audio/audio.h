/* audio — capture + DSP front-end for cPrompter (M1: voice mode).
 * Design (docs/audit/cprompter-01-plan.md §4.2, cprompter-02-deps.md):
 *   - zero external deps by default: ALSA via dlopen("libasound.so.2") so the
 *     binary still builds/runs everywhere; null backend for CI/headless tests.
 *   - everything downstream consumes mono float32 PCM @ 16 kHz via a callback
 *     running on the capture thread. VAD + WPM estimator attach there.
 */
#ifndef CP_AUDIO_H
#define CP_AUDIO_H

#include <stdint.h>
#include <stddef.h>

#define CP_AUDIO_RATE 16000   /* our internal working sample rate */

typedef struct cp_audio cp_audio;

/* frame: mono float32 in [-1,1], n samples at CP_AUDIO_RATE */
typedef void (*cp_audio_cb)(const float *pcm, size_t n, void *user);

typedef struct {
    const char *dev;        /* NULL = default device ("null" for CI)   */
    int         frame_ms;   /* callback granularity, e.g. 20           */
    cp_audio_cb cb;
    void       *user;
} cp_audio_params;

/* returns NULL on failure; err gets a message (may be NULL) */
cp_audio *cp_audio_open(const cp_audio_params *p, char *err, size_t errn);
int       cp_audio_start(cp_audio *a);   /* spawns capture thread */
void      cp_audio_stop(cp_audio *a);    /* stops and joins       */
void      cp_audio_close(cp_audio *a);

/* measured live input level 0..1 (peak of recent frames), for UI waveform */
float     cp_audio_level(cp_audio *a);

/* ---- DSP helpers (pure, testable, no globals) ---- */

/* downsample int16 src_rate -> CP_AUDIO_RATE with averaging pre-filter.
 * dst must hold >= n_in*CP_AUDIO_RATE/src_rate + 2 samples. Returns count. */
size_t cp_resample_i16_to16k(const int16_t *src, size_t n_in, int src_rate,
                             float *dst, size_t dst_cap);

/* short-time RMS over n samples (float pcm) */
float  cp_rms(const float *pcm, size_t n);

/* simple one-pole high-pass to tame rumble/breath noise; state persists */
typedef struct { float x1, y1; } cp_hp1;
void   cp_hp1_reset(cp_hp1 *h);
void   cp_hp1_run(cp_hp1 *h, float *pcm, size_t n);

#endif /* CP_AUDIO_H */
