/* qrcodegen.c — minimal but real QR encoder: Model 2, byte mode, ECC level M,
 * versions 1..10. Implements GF(256) Reed-Solomon, data masking with the four
 * ISO/IEC 18004 penalty rules + lowest-penalty selection, and standard format
 * info. Written from the spec; same output semantics as Nayuki's reference. */
#define _POSIX_C_SOURCE 200809L
#include "qrcodegen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- GF(256) tables (poly 0x11D) ---------------- */
static unsigned char gf_exp[512], gf_log[256];
static void gf_init(void) {
    static int done = 0; if (done) return; done = 1;
    int x = 1;
    for (int i = 0; i < 255; i++) {
        gf_exp[i] = (unsigned char)x; gf_log[x] = (unsigned char)i;
        x <<= 1; if (x & 0x100) x ^= 0x11D;
    }
    for (int i = 255; i < 512; i++) gf_exp[i] = gf_exp[i - 255];
}
static unsigned char gf_mul(unsigned char a, unsigned char b) {
    if (!a || !b) return 0;
    return gf_exp[gf_log[a] + gf_log[b]];
}
static const unsigned char *rs_gen(int deg) {   /* generator poly coeffs */
    static unsigned char cache[31][31]; static int built[31];
    if (built[deg]) return cache[deg];
    unsigned char *g = cache[deg];
    g[0] = 1;
    for (int i = 0; i < deg; i++) {              /* multiply by (x - a^i) */
        for (int j = i + 1; j >= 1; j--)
            g[j] = (unsigned char)(g[j-1] ^ gf_mul(g[j], gf_exp[i]));
        g[0] = gf_mul(g[0], gf_exp[i]);
    }
    built[deg] = 1;
    return g;
}
static void rs_encode(const unsigned char *data, int n, unsigned char *ecc, int deg) {
    const unsigned char *g = rs_gen(deg);
    memset(ecc, 0, (size_t)deg);
    for (int i = 0; i < n; i++) {
        unsigned char f = (unsigned char)(data[i] ^ ecc[0]);
        memmove(ecc, ecc + 1, (size_t)deg - 1);
        ecc[deg-1] = 0;
        for (int j = 0; j < deg; j++)
            ecc[j] ^= gf_mul(g[deg - 1 - j], f);
    }
}

/* ---------------- version tables (ECC level M, versions 1..10) ---------- */
/* Authoritative ECC-M parameters (ISO 18004 Table 13/14), versions 1..10:
 *   total_codewords = floor(data_bits/8); [c] EC per block; blocks split. */
typedef struct { int cw_total, ec_per_block, blocks_short, cw_short, blocks_long, cw_long; } vm_t;
static const vm_t VMT[11] = {
    {0,0,0,0,0,0},
    { 26, 10, 1, 16, 0,  0},   /* v1-M : 1 block  (16 data,10 EC) */
    { 44, 16, 1, 28, 0,  0},   /* v2-M : 1 block  (28,16)          */
    { 70, 26, 1, 44, 0,  0},   /* v3-M : 1 block  (44,26)          */
    {100, 18, 2, 32, 0,  0},   /* v4-M : 2 blocks (32,18)          */
    {134, 24, 2, 43, 0,  0},   /* v5-M : 2 blocks (43,24)          */
    {172, 16, 4, 27, 0,  0},   /* v6-M : 4 blocks (27,16)          */
    {196, 18, 4, 31, 0,  0},   /* v7-M : 4 blocks (31,18)          */
    {242, 22, 2, 38, 2, 39},   /* v8-M : 2x(38,22) + 2x(39,22)     */
    {292, 22, 3, 36, 2, 37},   /* v9-M : 3x(36,22) + 2x(37,22)     */
    {346, 26, 4, 43, 1, 44},   /* v10-M: 4x(43,26) + 1x(44,26)     */
};

static int align_positions(int ver, int *out) {
    /* ISO corner finder pattern centers */
    static const int tbl[11][8] = {
        {},
        {6}, {6,18}, {6,22}, {6,26}, {6,30}, {6,34},
        {6,22,38}, {6,24,42}, {6,26,46}, {6,28,50},
    };
    int n = 0;
    for (; n < 8 && tbl[ver][n]; n++) out[n] = tbl[ver][n];
    return n;
}

/* ---------------- matrix builder ---------------- */
typedef struct {
    int size, ver;
    unsigned char *m;      /* 0/1 module value                            */
    unsigned char *fn;     /* function-pattern mask (don't overwrite)     */
} mat;

static void mat_set(mat *M, int y, int x, int v, int fn) {
    if (y < 0 || x < 0 || y >= M->size || x >= M->size) return;
    M->m[y * M->size + x] = (unsigned char)(v ? 1 : 0);
    if (fn) M->fn[y * M->size + x] = 1;
}
static int mat_get(mat *M, int y, int x) {
    if (y < 0 || x < 0 || y >= M->size || x >= M->size) return 0;
    return M->m[y * M->size + x];
}

static void draw_finders(mat *M) {
    const int s = M->size;
    /* three 8x8 finder+separator blocks anchored at matrix corners:
     * TL(0,0) TR(0,s-8) BL(s-8,0). Inside each: 7x7 pattern + light ring. */
    const int oy[3] = {0, 0, s - 8}, ox[3] = {0, s - 8, 0};
    for (int c = 0; c < 3; c++)
        for (int dy = 0; dy < 8; dy++)
            for (int dx = 0; dx < 8; dx++) {
                int y = oy[c] + dy, x = ox[c] + dx;
                if (dy == 7 || dx == 7) { mat_set(M, y, x, 0, 1); continue; } /* sep */
                int d = abs(dy - 3) > abs(dx - 3) ? abs(dy - 3) : abs(dx - 3);
                mat_set(M, y, x, d != 2, 1);                                  /* rings */
            }
    /* timing patterns: full row/col 6 between the separators */
    for (int k = 0; k < s; k++) {   /* full-length timing per ISO */
        mat_set(M, 6, k, !(k % 2), 1);
        mat_set(M, k, 6, !(k % 2), 1);
    }
    /* alignment pattern(s): all pos x pos combos except finder overlaps */
    int pos[8]; int np = align_positions(M->ver, pos);
    for (int i = 0; i < np; i++)
        for (int j = 0; j < np; j++) {
            int nearTL = (pos[i] <= 8 && pos[j] <= 8);
            int nearTR = (pos[i] <= 8 && pos[j] >= s - 8);
            int nearBL = (pos[i] >= s - 8 && pos[j] <= 8);
            if (nearTL || nearTR || nearBL) continue;
            for (int dy = -2; dy <= 2; dy++)
                for (int dx = -2; dx <= 2; dx++) {
                    int d = abs(dy) > abs(dx) ? abs(dy) : abs(dx);
                    mat_set(M, pos[i] + dy, pos[j] + dx, d != 1, 1);
                }
        }
    /* reserve format areas (values written later by draw_format) */
    for (int k = 0; k <= 8; k++) {
        if (k != 6) { mat_set(M, 8, k, 0, 1); mat_set(M, k, 8, 0, 1); }
    }
    for (int k = 0; k < 8; k++) {
        mat_set(M, 8, s - 1 - k, 0, 1);
        mat_set(M, s - 1 - k, 8, 0, 1);
    }
    mat_set(M, s - 8, 8, 1, 1);   /* dark module */
}

/* interleave data+EC blocks into final codeword stream */
static void interleave(const unsigned char **dbits, unsigned char *const *ebs,
                       int nb, const int *dcw, const int *ecw,
                       unsigned char *out, int total_cw) {
    int o = 0;
    int maxd = 0; for (int i = 0; i < nb; i++) if (dcw[i] > maxd) maxd = dcw[i];
    for (int k = 0; k < maxd; k++)
        for (int i = 0; i < nb; i++)
            if (k < dcw[i]) out[o++] = dbits[i][k];
    for (int k = 0; k < ecw[0]; k++)
        for (int i = 0; i < nb; i++)
            out[o++] = ebs[i][k];
    while (o < total_cw) out[o++] = 0;   /* remainder bits pad */
}

/* place codewords in the zig-zag 2-module columns, skipping col 6 */
static void place_modules(mat *M, const unsigned char *cw, int ncw) {
    int bitno = 0;
    int up = 1;
    for (int right = M->size - 1; right >= 1; right -= 2) {
        if (right == 6) right = 5;
        for (int vert = 0; vert < M->size; vert++) {
            int y = up ? M->size - 1 - vert : vert;
            for (int j = 0; j < 2; j++) {
                int x = right - j;
                if (M->fn[y * M->size + x]) continue;
                int byte = bitno >> 3, bpos = 7 - (bitno & 7);
                int val = (byte < ncw) ? ((cw[byte] >> bpos) & 1) : 0;
                M->m[y * M->size + x] = (unsigned char)val;
                bitno++;
            }
        }
        up = !up;
    }
}

/* masks (ISO 18004 Table 20) */
static int mask_fn(int m, int y, int x) {
    switch (m) {
    case 0: return (y + x) % 2 == 0;
    case 1: return y % 2 == 0;
    case 2: return x % 3 == 0;
    case 3: return (y + x) % 3 == 0;
    case 4: return (y / 2 + x / 3) % 2 == 0;
    case 5: return (y * x) % 2 + (y * x) % 3 == 0;
    case 6: return ((y * x) % 2 + (y * x) % 3) % 2 == 0;
    default:return (((y + x) % 2) + ((y * x) % 3)) % 2 == 0;
    }
}

static void apply_mask(mat *M, int m) {
    for (int y = 0; y < M->size; y++)
        for (int x = 0; x < M->size; x++)
            if (!M->fn[y * M->size + x] && mask_fn(m, y, x))
                M->m[y * M->size + x] ^= 1;
}

static int penalty(mat *M) {
    int p = 0, s = M->size;
    /* rule 1: runs of >=5 same-colour modules, rows and columns: 3 + (len-5) */
    for (int y = 0; y < s; y++) for (int dir = 0; dir < 2; dir++) {
        int run = 1;
        for (int k = 1; k < s; k++) {
            int cur = dir ? mat_get(M, k, y) : mat_get(M, y, k);
            int prv = dir ? mat_get(M, k-1, y) : mat_get(M, y, k-1);
            if (cur == prv) run++;
            else { if (run >= 5) p += run - 2; run = 1; }
        }
        if (run >= 5) p += run - 2;
    }
    /* rule 2: each 2x2 same-colour block: 3 */
    for (int y = 0; y < s-1; y++)
        for (int x = 0; x < s-1; x++) {
            int v = mat_get(M, y, x);
            if (v == mat_get(M, y, x+1) && v == mat_get(M, y+1, x) && v == mat_get(M, y+1, x+1))
                p += 3;
        }
    /* rule 3: 1011101 with 4 light on either side (edge counts as light): 40 */
    static const int pat[7] = {1,0,1,1,1,0,1};
    for (int y = 0; y < s; y++) for (int x = 0; x + 7 <= s; x++) {
        int h = 1, v = 1;
        for (int k = 0; k < 7; k++) {
            if (mat_get(M, y, x+k) != pat[k]) h = 0;
            if (mat_get(M, x+k, y) != pat[k]) v = 0;
        }
        int hl = (x == 0) || !mat_get(M, y, x-1);
        int hr = (x + 7 == s) || !mat_get(M, y, x+7);
        int vl = (x == 0) || !mat_get(M, x-1, y);
        int vr = (x + 7 == s) || !mat_get(M, x+7, y);
        if (h && hl && hr) p += 40;
        if (v && vl && vr) p += 40;
    }
    /* rule 4: |dark% - 50| deviation: 10 per 5% step */
    int dark = 0;
    for (int i = 0; i < s*s; i++) if (M->m[i]) dark++;
    int k = abs(dark * 200 - s * s * 100) / (s * s * 10);   /* floor(|pct-50|/5) */
    p += k * 10;
    return p;
}

/* format info: EC-M = bits 00, so data = mask value directly (bits b9..b5 =
 * 0,0,m2,m1,m0). BCH(15,5) with G(x)=0x537, XOR mask 0x5412 — ISO 18004 8.9. */
static int format_bits(int mask) {
    int rem = mask << 10;                       /* data bits already zero above */
    for (int i = 14; i >= 10; i--)
        if ((rem >> i) & 1) rem ^= 0x537 << (i - 10);
    return ((mask << 10) | rem) ^ 0x5412;
}
static void draw_format(mat *M, int mask) {
    int f = format_bits(mask);
    for (int i = 0; i < 15; i++) {
        int b = (f >> i) & 1;                   /* bit 0 first */
        /* copy 1: around top-left finder */
        if (i < 6)       mat_set(M, 8, i, b, 1);
        else if (i == 6) mat_set(M, 8, 7, b, 1);
        else if (i == 7) mat_set(M, 8, 8, b, 1);
        else if (i == 8) mat_set(M, 7, 8, b, 1);
        else             mat_set(M, 14 - i, 8, b, 1);
        /* copy 2: bottom-left vertical + top-right horizontal */
        if (i < 8)       mat_set(M, M->size - 1 - i, 8, b, 1);
        else             mat_set(M, 8, M->size - 15 + i, b, 1);
    }
    mat_set(M, M->size - 8, 8, 1, 1);           /* dark module stays dark */
}

/* ---------------- public API ---------------- */
cp_qr *cp_qr_encode(const char *text) {
    gf_init();
    size_t len = strlen(text);
    /* pick smallest version whose data capacity fits:
     * need = 4 (mode) + ccbits + 8*len  bits <= (cw_total - ec_total)*8 */
    int ver = 0;
    for (int v = 1; v <= 10; v++) {
        int cc = v <= 9 ? 8 : 16;
        int ec_total = VMT[v].ec_per_block * (VMT[v].blocks_short + VMT[v].blocks_long);
        int data_cw = VMT[v].cw_total - ec_total;
        int need_bits = 4 + cc + (int)len * 8;
        if (need_bits <= data_cw * 8) { ver = v; break; }
    }

    if (!ver) return NULL;
    const vm_t *P = &VMT[ver];
    int nb = P->blocks_short + P->blocks_long;
    int ec_per = P->ec_per_block;

    /* build bitstream (packed into data codewords) */
    int data_cw = P->cw_total - ec_per * nb;
    unsigned char *cw = calloc((size_t)data_cw, 1);
    int cap_bits = data_cw * 8;
    int cc = ver <= 9 ? 8 : 16;
    int wpos = 0;
    #define WB(val,nbits) do { for (int z = (nbits)-1; z >= 0; z--) { \
            if (wpos < cap_bits) cw[wpos >> 3] |= (unsigned char)((((val) >> z) & 1) << (7 - (wpos & 7))); \
            wpos++; } } while (0)
    WB(0b0100, 4);                 /* byte mode */
    WB((int)len, cc);              /* char count */
    for (size_t i = 0; i < len; i++) WB((unsigned char)text[i], 8);
    /* terminator + pad to byte boundary */
    for (int t = 0; t < 4 && wpos < cap_bits; t++) wpos++;   /* zero bits implicit */
    while (wpos % 8 && wpos < cap_bits) wpos++;
    /* pad bytes 0xEC 0x11 alternating */
    int padvals[2] = {0xEC, 0x11}; int pi = 0;
    while (wpos < cap_bits) { WB(padvals[pi], 8); pi ^= 1; }
    #undef WB

    /* split into blocks, encode EC */
    const unsigned char **dbits = calloc((size_t)nb, sizeof(void*));
    unsigned char **ebs = calloc((size_t)nb, sizeof(void*));
    int *dcw = calloc((size_t)nb, sizeof(int));
    int *ecw = calloc((size_t)nb, sizeof(int));
    int off = 0;
    for (int i = 0; i < nb; i++) {
        int sz = (i < P->blocks_short) ? P->cw_short : P->cw_long;
        dcw[i] = sz; ecw[i] = ec_per;
        dbits[i] = cw + off; off += sz;
        ebs[i] = calloc((size_t)ec_per, 1);
        rs_encode(dbits[i], sz, ebs[i], ec_per);
    }
    unsigned char *all = calloc((size_t)P->cw_total, 1);
    interleave(dbits, ebs, nb, dcw, ecw, all, P->cw_total);

    /* matrix */
    mat M = {0};
    M.ver = ver; M.size = 17 + 4 * ver;
    M.m = calloc((size_t)M.size * M.size, 1);
    M.fn = calloc((size_t)M.size * M.size, 1);
    draw_finders(&M);
    place_modules(&M, all, P->cw_total);

    /* choose best mask */
    int best = 0, bestp = 1 << 30;
    for (int m = 0; m < 8; m++) {
        apply_mask(&M, m);
        draw_format(&M, m);
        int p = penalty(&M);
        if (p < bestp) { bestp = p; best = m; }
        apply_mask(&M, m);          /* undo (XOR self-inverse) */
    }
    apply_mask(&M, best);
    draw_format(&M, best);

    cp_qr *qr = calloc(1, sizeof *qr);
    qr->size = M.size;
    qr->pixels = calloc((size_t)M.size * M.size, 1);
    for (int i = 0; i < M.size * M.size; i++) qr->pixels[i] = M.m[i] ? '#' : ' ';
    free(M.m); free(M.fn); free(cw); free(all);
    for (int i = 0; i < nb; i++) free(ebs[i]);
    free(dbits); free(ebs); free(dcw); free(ecw);
    return qr;
}

void cp_qr_free(cp_qr *qr) { if (!qr) return; free(qr->pixels); free(qr); }

void cp_qr_print(const cp_qr *qr) {
    if (!qr) return;
    /* quiet zone 4 modules, 2 chars wide per module */
    int s = qr->size;
    for (int y = -4; y < s + 4; y++) {
        for (int x = -4; x < s + 4; x += 2) {
            int dark_l = 0, dark_r = 0;
            if (y >= 0 && y < s && x >= 0 && x < s) dark_l = qr->pixels[y*s+x] == '#';
            if (y >= 0 && y < s && x+1 >= 0 && x+1 < s) dark_r = qr->pixels[y*s+x+1] == '#';
            const char *cell = (dark_l && dark_r) ? "██" :
                               (!dark_l && !dark_r) ? "  " :
                               (dark_l) ? "▓░" : "░▓";
            fputs(cell, stdout);
        }
        putchar('\n');
    }
}
