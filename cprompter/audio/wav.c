/* wav.c — minimal RIFF/PCM + IEEE-float WAV reader for tests & offline runs.
 * Feeds the same cp_audio_cb contract as live capture, so vad/wpm/align code
 * is testable with recorded speech without any mic. (~120 lines.) */
#define _GNU_SOURCE
#include "wav.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "audio.h"   /* CP_AUDIO_RATE, cp_resample_i16_to16k */

int cp_wav_write_i16(const char *path, const int16_t *pcm, size_t n, int rate) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t data_bytes = (uint32_t)(n * 2);
    uint32_t riff = 36 + data_bytes, sr = (uint32_t)rate;
    uint16_t ch = 1, bits = 16, ba = 2;
    unsigned char h[44];
    memcpy(h, "RIFF", 4);
    h[4] = riff & 0xff; h[5] = riff >> 8 & 0xff; h[6] = riff >> 16 & 0xff; h[7] = riff >> 24;
    memcpy(h + 8, "WAVEfmt ", 8);
    h[16] = 16; h[17] = h[18] = h[19] = 0;
    h[20] = 1; h[21] = 0;                 /* PCM            */
    h[22] = ch; h[23] = 0;                /* mono           */
    h[24] = sr & 0xff; h[25] = sr >> 8 & 0xff; h[26] = sr >> 16 & 0xff; h[27] = sr >> 24;
    uint32_t brate = sr * ba;
    h[28] = brate & 0xff; h[29] = brate >> 8 & 0xff; h[30] = brate >> 16 & 0xff; h[31] = brate >> 24;
    h[32] = ba; h[33] = 0;
    h[34] = bits; h[35] = 0;
    memcpy(h + 36, "data", 4);
    h[40] = data_bytes & 0xff; h[41] = data_bytes >> 8 & 0xff;
    h[42] = data_bytes >> 16 & 0xff; h[43] = data_bytes >> 24;
    fwrite(h, 1, 44, f);
    fwrite(pcm, 2, n, f);
    fclose(f);
    return 0;
}

typedef struct {
    int      rate;
    int      channels;
    int      bits;          /* 16 or 32 (float) */
    int      is_float;
    size_t   nsamp;         /* frames */
    float   *pcm;           /* mono float @ original rate             */
} wavfile;

#ifdef CP_WAV_MAIN
static uint32_t rd32(const unsigned char *p) { return p[0]|p[1]<<8|p[2]<<16|(uint32_t)p[3]<<24; }
static uint16_t rd16(const unsigned char *p) { return p[0]|p[1]<<8; }

/* wav_read is only used by the offline CLI smoke test below */
static int wav_read(const char *path, wavfile *o) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long flen = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *buf = malloc(flen);
    if (!buf || fread(buf, 1, flen, f) != (size_t)flen) { free(buf); fclose(f); return -1; }
    fclose(f);
    memset(o, 0, sizeof *o);
    int rc = -1;
    if (flen < 44 || memcmp(buf, "RIFF", 4) || memcmp(buf + 8, "WAVE", 4)) goto done;

    long pos = 12;
    const unsigned char *data = NULL; size_t dlen = 0;
    while (pos + 8 <= flen) {
        uint32_t csz = rd32(buf + pos + 4);
        if (!memcmp(buf + pos, "fmt ", 4) && csz >= 16) {
            o->channels = rd16(buf + pos + 10);
            o->rate     = (int)rd32(buf + pos + 12);
            o->bits     = rd16(buf + pos + 22);
            o->is_float = (rd16(buf + pos + 8) == 3 /*IEEE float*/);
        } else if (!memcmp(buf + pos, "data", 4)) {
            data = buf + pos + 8; dlen = csz;
            if ((long)(pos + 8 + csz) > flen) dlen = flen - pos - 8;
        }
        pos += 8 + csz + (csz & 1);
    }
    if (!data || o->rate <= 0 || o->channels <= 0) goto done;
    if (o->bits != 16 && !(o->bits == 32 && o->is_float)) goto done;

    size_t bytes_per_frame = o->channels * o->bits / 8;
    o->nsamp = dlen / bytes_per_frame;
    o->pcm = malloc(o->nsamp * sizeof(float));
    if (!o->pcm) goto done;
    for (size_t i = 0; i < o->nsamp; i++) {
        const unsigned char *p = data + i * bytes_per_frame;
        double acc = 0.0;
        for (int c = 0; c < o->channels; c++) {
            if (o->bits == 16) acc += (int16_t)rd16(p + c * 2) / 32768.0;
            else { uint32_t u = rd32(p + c * 4); float fv; memcpy(&fv, &u, 4); acc += fv; }
        }
        o->pcm[i] = (float)(acc / o->channels);
    }
    rc = 0;
done:
    if (rc != 0) free(o->pcm);
    free(buf);
    return rc;
}
#endif /* CP_WAV_MAIN */

/* CLI: ./wavtest in.wav -> prints detected speech segments via VAD,
 * and WPM estimate. Used by `make test` as a synthetic-speech smoke test. */
#ifdef CP_WAV_MAIN
#include "vad.h"
#include "wpm.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.wav\n", argv[0]); return 2; }
    wavfile w;
    if (wav_read(argv[1], &w) != 0) { fprintf(stderr, "bad wav\n"); return 1; }
    cp_vad *v = cp_vad_new(w.rate);
    cp_wpm *m = cp_wpm_new(w.rate);
    float *pcm = w.pcm; size_t n = w.nsamp; int rate = w.rate;
    /* VAD/WPM are rate-parameterized, so we analyze at the file's own rate.
     * (cp_resample_i16_to16k is used by live capture when the device differs.) */
    int seg = 0, was = 0; double t0 = 0, dur = 0;
    size_t step = (size_t)rate / 50;   /* 20 ms */
    for (size_t i = 0; i + step <= n; i += step) {
        int sp = cp_vad_push(v, pcm + i, step);
        cp_wpm_push(m, pcm + i, step, sp);
        if (sp && !was) { seg++; t0 = (double)i / rate; }
        if (!sp && was) dur += (double)i / rate - t0;
        was = sp;
    }
    printf("segments=%d speech=%.2fs wpm_est=%.0f syllables=%d\n",
           seg, dur, cp_wpm_estimate(m), cp_wpm_syllables_total(m));
    cp_vad_free(v); cp_wpm_free(m); free(w.pcm);
    return 0;
}
#endif
