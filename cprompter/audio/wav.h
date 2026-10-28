/* wav.h — tiny WAV writer for tests (pairs with the reader in wav.c). */
#ifndef CP_WAV_H
#define CP_WAV_H

#include <stddef.h>
#include <stdint.h>

/* write mono int16 PCM as 44-byte RIFF WAV. returns 0 on success */
int cp_wav_write_i16(const char *path, const int16_t *pcm, size_t n, int rate);

#endif
