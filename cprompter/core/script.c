/* script.c — tokenizer & page model. UTF-8 safe: words are byte ranges split
 * on Unicode-ish whitespace; punctuation stays attached to tokens (matches
 * how whisper output is later aligned in align.c). */
#define _POSIX_C_SOURCE 200809L
#include "script.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ---- helpers ------------------------------------------------------------ */

static int is_space_byte(unsigned char c) {
    /* ASCII space set + NBSP/UTF-8 continuation handled below via lead bytes */
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f';
}

/* returns 1 if s[i..] starts a UTF-8 whitespace codepoint we care about */
static int is_space_utf8(const char *s, size_t i, size_t *adv) {
    unsigned char c = (unsigned char)s[i];
    if (is_space_byte(c)) { *adv = 1; return 1; }
    /* U+00A0 NBSP (C2 A0), U+2000-200B en/em spaces etc. (E2 80 8x-9x / B),
     * U+3000 ideographic (E3 80 80), U+202F narrow NBSP (E2 80 AF) */
    if (c == 0xC2 && (unsigned char)s[i+1] == 0xA0) { *adv = 2; return 1; }
    if (c == 0xE2 && (unsigned char)s[i+1] == 0x80) {
        unsigned char t = (unsigned char)s[i+2];
        if ((t >= 0x80 && t <= 0x8B) || t == 0x8F || t == 0x9F) { *adv = 3; return 1; }
    }
    if (c == 0xE3 && (unsigned char)s[i+1] == 0x80 && (unsigned char)s[i+2] == 0x80) {
        *adv = 3; return 1;
    }
    *adv = 0;
    return 0;
}

static void words_push(cp_script *s, const char *text, size_t n, int para) {
    if (s->nw == s->capw) {
        size_t nc = s->capw ? s->capw * 2 : 512;
        cp_word *w = realloc(s->words, nc * sizeof *w);
        if (!w) return;
        s->words = w; s->capw = nc;
    }
    cp_word *w = &s->words[s->nw++];
    w->text = malloc(n + 1);
    if (!w->text) { s->nw--; return; }
    memcpy(w->text, text, n);
    w->text[n] = 0;
    w->n = n;
    w->para = para;
    w->wpm_prior = 1.0f;
}

static void pages_push(cp_script *s, const char *name, int w0) {
    if (s->np == s->capp) {
        size_t nc = s->capp ? s->capp * 2 : 16;
        cp_page *p = realloc(s->pages, nc * sizeof *p);
        if (!p) return;
        s->pages = p; s->capp = nc;
    }
    cp_page *p = &s->pages[s->np++];
    memset(p, 0, sizeof *p);
    snprintf(p->name, sizeof p->name, "%s", name && *name ? name : "Page");
    p->w0 = w0; p->w1 = w0;
    p->mode = (cp_mode)-1;          /* -1 => inherit default */
    p->scroll_speed = 0;            /* 0 => inherit default  */
    p->countdown_sec = 3;           /* Textream's default    */
}

/* ---- public API --------------------------------------------------------- */

cp_script *cp_script_new(void) {
    cp_script *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->default_mode  = CP_MODE_CLASSIC;
    s->default_speed = 3.0;             /* blog: classic default ~3 w/s */
    snprintf(s->lang, sizeof s->lang, "en-US");
    s->font_size = 24;
    return s;
}

void cp_script_free(cp_script *s) {
    if (!s) return;
    for (size_t i = 0; i < s->nw; i++) free(s->words[i].text);
    free(s->words);
    free(s->pages);
    free(s->raw);
    free(s);
}

int cp_script_load_text(cp_script *s, const char *text, int split_pages) {
    if (!s || !text) return -1;
    /* reset previous content */
    for (size_t i = 0; i < s->nw; i++) free(s->words[i].text);
    s->nw = 0; s->np = 0;
    free(s->raw);
    s->raw = strdup(text);
    if (!s->raw) return -1;

    int para = 0;
    int new_page = 1;               /* start first page */
    char page_name[CP_MAX_PAGE_NAME] = {0};
    size_t i = 0;
    size_t len = strlen(text);

    /* file may begin directly with a "# Title" line (no preceding \n): */
    if (split_pages && len >= 2 && text[0] == '#' && text[1] == ' ') {
        size_t k = 2, st = 2;
        while (k < len && text[k] != '\n') k++;
        size_t cn = k - st; if (cn >= sizeof page_name) cn = sizeof page_name - 1;
        memcpy(page_name, text + st, cn); page_name[cn] = 0;
        i = k;                       /* skip heading; page opens at next word */
        para++;
    }

    while (i < len) {
        size_t adv;
        /* line-level structure: blank lines end a paragraph; "# " starts page */
        if (text[i] == '\n') {
            /* detect blank-line paragraph break */
            size_t j = i + 1;
            while (j < len && (text[j] == ' ' || text[j] == '\t' || text[j] == '\r')) j++;
            if (j < len && text[j] == '\n') { para++; }   /* blank line -> new para */
            /* heading? "# Title" begins a page when split_pages */
            if (split_pages && j + 1 < len && text[j] == '#' && text[j+1] == ' ') {
                size_t k = j + 2, st = k;
                while (k < len && text[k] != '\n') k++;
                size_t cn = k - st; if (cn >= sizeof page_name) cn = sizeof page_name - 1;
                memcpy(page_name, text + st, cn); page_name[cn] = 0;
                new_page = 1;
                para++;
                i = k;
                continue;
            }
            i++;
            continue;
        }
        if (is_space_utf8(text, i, &adv)) { i += adv; continue; }

        /* word = run of non-space bytes */
        size_t st = i;
        while (i < len) {
            size_t a2;
            if (is_space_utf8(text, i, &a2)) break;
            unsigned char c = (unsigned char)text[i];
            i += (c < 0x80) ? 1 : ((c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : 2);
        }
        if (new_page) {
            pages_push(s, page_name[0] ? page_name : "", (int)s->nw);
            page_name[0] = 0;
            new_page = 0;
        }
        words_push(s, text + st, i - st, para);
    }
    if (s->np == 0) pages_push(s, "Script", 0);   /* empty input still valid */
    /* close page ranges */
    for (size_t p = 0; p < s->np; p++)
        s->pages[p].w1 = (p + 1 < s->np) ? s->pages[p + 1].w0 : (int)s->nw;
    return 0;
}

const cp_word *cp_word_at(const cp_script *s, size_t i) {
    return i < s->nw ? &s->words[i] : NULL;
}

int cp_page_of(const cp_script *s, size_t word_idx) {
    for (size_t p = 0; p < s->np; p++)
        if ((int)word_idx >= s->pages[p].w0 && (int)word_idx < s->pages[p].w1)
            return (int)p;
    return (int)(s->np ? s->np - 1 : 0);
}

size_t cp_script_count_words(const cp_script *s) { return s->nw; }

double cp_script_est_seconds(const cp_script *s, double wps) {
    if (!s || wps <= 0) return 0;
    double tot = 0;
    for (size_t i = 0; i < s->nw; i++) tot += s->words[i].wpm_prior;
    return tot / wps;
}
