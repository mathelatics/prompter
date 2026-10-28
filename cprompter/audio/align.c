/* align.c — see align.h. Deterministic, allocation-light word tracker. */
#define _POSIX_C_SOURCE 200809L
#include "align.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define AL_MAX_LOOK   12   /* how far ahead we search for skipped words      */
#define AL_MAX_BACK    4   /* how far back re-reads may move the cursor      */
#define AL_GIVEUP     30   /* consecutive misses before drift flag           */
#define AL_WINDOW_MIN  2   /* min fuzzy window (short words need exactness)  */

struct cp_align {
    const cp_script *s;
    size_t  idx;        /* current matched script word index                 */
    int     drift;      /* miss streak                                       */
    char    last[64];   /* last heard normalized word (UI ticker)            */
    /* normalized cache of script words: lazily built once per script */
    char   *norm;       /* flat buffer, AL_NORMSZ bytes per word             */
    size_t  norm_n;     /* number of entries cached                          */
};

#define AL_NORMSZ 48

static void ensure_norm(cp_align *a, size_t upto) {
    if (upto > a->s->nw) upto = a->s->nw;
    if (upto <= a->norm_n) return;
    size_t ncap = a->s->nw;
    if (!a->norm) a->norm = calloc(ncap, AL_NORMSZ);
    for (size_t i = a->norm_n; i < upto && a->norm; i++)
        cp_norm_word(a->s->words[i].text, a->norm + i * AL_NORMSZ, AL_NORMSZ);
    a->norm_n = upto;
}

cp_align *cp_align_new(const cp_script *s) {
    cp_align *a = calloc(1, sizeof *a);
    a->s = s;
    ensure_norm(a, s->nw < 64 ? s->nw : 64);
    return a;
}

void cp_align_free(cp_align *a) {
    if (!a) return;
    free(a->norm);
    free(a);
}

void cp_align_reset(cp_align *a, size_t start) {
    if (start > a->s->nw) start = a->s->nw ? a->s->nw - 1 : 0;
    a->idx = start;
    a->drift = 0;
}

size_t cp_align_pos(const cp_align *a) { return a->idx; }
int    cp_align_drift(const cp_align *a) { return a->drift; }
const char *cp_align_last_heard(const cp_align *a) { return a->last; }

/* ---- normalization: lowercase ascii, strip punctuation/quotes/apostrophes.
 * UTF-8 continuation bytes are passed through so accented words still match
 * each other byte-wise; only ASCII junk is removed. */
int cp_norm_word(const char *in, char *out, size_t outn) {
    size_t j = 0;
    for (const unsigned char *p = (const unsigned char *)in; *p; p++) {
        unsigned char c = *p;
        if (c == '\'' || c == '`' || c == 0xE2) continue; /* curly quote lead */
        if (isalnum(c) || c >= 0x80) {
            if (j + 1 < outn) out[j++] = (char)tolower(c);
        }
    }
    out[j] = 0;
    return (int)j;
}

static const char *FILLERS[] = {
    "um","uh","erm","er","ah","oh","mm","hmm",
    "like","so","well","okay","ok","yeah","you","really","basically",
    "i","and","but","the","a","an","to","it","this","that", NULL
};
/* NOTE: common function words are listed as "soft fillers": they are only
 * skipped when the *next expected* word doesn't match and nothing in the
 * lookahead matches either — handled in feed(), not blindly dropped. */

int cp_is_filler(const char *norm) {
    static const char *hard[] = { "um","uh","erm","er","ah","oh","mm","hmm", NULL };
    for (int i = 0; hard[i]; i++) if (!strcmp(norm, hard[i])) return 1;
    return 0;
}

static int is_soft_filler(const char *norm) {
    for (int i = 0; FILLERS[i]; i++) if (!strcmp(FILLERS[i], norm)) return 1;
    return 0;
}

/* Levenshtein with early exit at dist (tiny arrays, stack only) */
static int lev_le(const char *a, const char *b, int dist) {
    int la = (int)strlen(a), lb = (int)strlen(b);
    if (abs(la - lb) > dist) return 0;
    if (la > 32 || lb > 32) { /* fallback: prefix rule only */
        int m = la < lb ? la : lb;
        return m >= 4 && !memcmp(a, b, m);
    }
    int prev[34], cur[34];
    for (int j = 0; j <= lb; j++) prev[j] = j;
    for (int i = 1; i <= la; i++) {
        cur[0] = i;
        int rowmin = cur[0];
        for (int j = 1; j <= lb; j++) {
            int cost = (a[i-1] == b[j-1]) ? 0 : 1;
            int d = prev[j-1] + cost;
            if (prev[j] + 1 < d) d = prev[j] + 1;
            if (cur[j-1] + 1 < d) d = cur[j-1] + 1;
            cur[j] = d;
            if (d < rowmin) rowmin = d;
        }
        if (rowmin > dist) return 0;
        memcpy(prev, cur, (lb + 1) * sizeof(int));
    }
    return prev[lb] <= dist;
}

int cp_fuzzy_eq(const char *a, const char *b) {
    if (!*a || !*b) return 0;
    /* long words: shared 4-char prefix counts (ASR truncations) */
    size_t la = strlen(a), lb = strlen(b);
    size_t m = la < lb ? la : lb;
    if (m >= 4 && !memcmp(a, b, 4) && (la >= 5 || lb >= 5)) {
        /* prefix match ok if one is near-prefix of other */
        if (lev_le(a, b, 2)) return 1;
    }
    int dl = (int)(la > lb ? la : lb);
    int allowed = dl <= 3 ? 0 : dl <= 6 ? 1 : 2;
    return lev_le(a, b, allowed);
}

/* try to match `w` against script[i]; returns 1 exact-norm, 1&fuzzy same code */
static int match_at(const cp_align *a, const char *w, size_t i) {
    const char *sw = a->norm + i * AL_NORMSZ;
    if (!*sw) return 0;
    if (!strcmp(w, sw)) return 1;
    return cp_fuzzy_eq(w, sw);
}

cp_align_result cp_align_feed(cp_align *a, const char *word) {
    cp_align_result r = { 0, 0, 0, a->idx, a->drift };
    char w[AL_NORMSZ];
    cp_norm_word(word, w, sizeof w);
    if (!w[0]) { r.script_idx = a->idx; return r; }
    snprintf(a->last, sizeof a->last, "%s", w);

    size_t nw = a->s->nw;
    if (nw == 0) return r;
    ensure_norm(a, (a->idx + AL_MAX_LOOK + 2) < nw ? a->idx + AL_MAX_LOOK + 2 : nw);

    size_t old = a->idx;

    /* 1) hard filler: drop silently, no drift penalty */
    if (cp_is_filler(w)) { r.filler_skipped = 1; r.script_idx = a->idx; return r; }

    /* 2) fast path: next expected word */
    if (a->idx < nw && match_at(a, w, a->idx)) {
        a->idx++; a->drift = 0;
        r.matched = 1; r.advanced = 1; r.script_idx = a->idx; r.drift = a->drift;
        return r;
    }

    /* 3) already-delivered re-read/stutter: step back within AL_MAX_BACK */
    if (a->idx > 0) {
        size_t lo = a->idx > (size_t)AL_MAX_BACK ? a->idx - AL_MAX_BACK : 0;
        for (size_t i = a->idx; i-- > lo;) {
            if (match_at(a, w, i)) {
                a->idx = i + 1; a->drift = 0;
                r.matched = 1; r.advanced = (int)a->idx - (int)old;
                r.script_idx = a->idx; r.drift = a->drift;
                return r;
            }
        }
    }

    /* 4) lookahead skip-ahead */
    size_t hi = a->idx + AL_MAX_LOOK < nw ? a->idx + AL_MAX_LOOK : nw;
    for (size_t i = a->idx + 1; i < hi; i++) {
        if (match_at(a, w, i)) {
            a->idx = i + 1; a->drift = 0;
            r.matched = 1; r.advanced = (int)a->idx - (int)old;
            r.script_idx = a->idx; r.drift = a->drift;
            return r;
        }
    }

    /* 5) soft filler / off-script word: ignore without drift penalty if we
     *    have recent momentum, else count a miss. */
    if (is_soft_filler(w) && a->drift < AL_GIVEUP) {
        r.filler_skipped = 1; r.script_idx = a->idx; return r;
    }
    a->drift++;
    r.script_idx = a->idx; r.drift = a->drift;
    return r;
}
