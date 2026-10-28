/* qrcodegen.h — minimal QR encoder (Model 2, byte mode, version auto-select,
 * ECC level M). Self-contained implementation in the spirit of Nayuki's
 * qrcodegen (MIT), written from the ISO/IEC 18004 spec for cPrompter's
 * "scan-to-view" remote/director URLs. No dependencies. */
#ifndef CP_QRCODEGEN_H
#define CP_QRCODEGEN_H

#include <stddef.h>

typedef struct {
    int   size;            /* modules per side (21..40)                      */
    char *pixels;          /* size*size bytes: '#' black, ' ' white          */
} cp_qr;

/* Encode `text` (UTF-8 bytes, <= ~700 chars at v10-M byte mode) into a QR
 * symbol. Returns NULL on failure (too long / allocation). Caller frees with
 * cp_qr_free(). */
cp_qr *cp_qr_encode(const char *text);
void   cp_qr_free(cp_qr *qr);

/* Print to stdout with 2-char-per-module blocks (terminal half-block safe):
 * uses "██"/"  " style via ANSI-free double spaces for light terminals. */
void cp_qr_print(const cp_qr *qr);

#endif /* CP_QRCODEGEN_H */
