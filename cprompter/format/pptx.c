/* pptx.c — M4: PowerPoint presenter-notes import (Textream feature parity).
 * Strategy (per docs/audit/cprompter-02-deps.md): use system `unzip` via popen to
 * extract notesSlideN.xml entries, then scan raw XML for <a:t> text runs. No full
 * OOXML parser needed: presenter notes are plain text runs inside <a:t>...</a:t>.
 * Each notes slide becomes one page; slide order = numeric order of part names. */
#define _POSIX_C_SOURCE 200809L   /* popen/pclose under -std=c11 */
#include "pptx.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXPAGES 256

typedef struct { int n; char *text; } note_t;

static int cmp_note(const void *a, const void *b) {
    return ((const note_t *)a)->n - ((const note_t *)b)->n;
}

/* XML entity decode in place; also strips nothing else. */
static void xml_unescape(char *s) {
    char *r = s;
    for (char *w = s; *w; ) {
        if (*w == '&') {
            if (!strncmp(w, "&amp;", 5))  { *r++ = '&'; w += 5; }
            else if (!strncmp(w, "&lt;", 4))   { *r++ = '<'; w += 4; }
            else if (!strncmp(w, "&gt;", 4))   { *r++ = '>'; w += 4; }
            else if (!strncmp(w, "&quot;", 6)) { *r++ = '"'; w += 6; }
            else if (!strncmp(w, "&apos;", 6)) { *r++ = '\''; w += 6; }
            else if (!strncmp(w, "&#10;", 5) || !strncmp(w, "&#x0A;", 6) ||
                       !strncmp(w, "&#xA;", 5))  { *r++ = '\n'; w += w[1]=='x'?6:5; }
            else if (!strncmp(w, "&#13;", 5))  { w += 5; } /* CR: drop */
            else *r++ = *w++;
        } else *r++ = *w++;
    }
    *r = 0;
}

/* Extract concatenated <a:t>...</a:t> contents from an XML buffer into out
 * (heap, NUL-terminated). Paragraph breaks: </a:p> boundaries -> '\n'.
 * Returns malloc'd string or NULL on OOM. */
char *cp_pptx_extract_at_texts(const char *xml, size_t len) {
    size_t cap = 256, n = 0;
    char *out = malloc(cap);
    if (!out) return NULL;
    const char *p = xml, *end = xml + len;
    while (p < end) {
        const char *t = strstr(p, "<a:t");
        if (!t || t > end - 4) break;
        /* must be followed by '>' or space (attribute), not e.g. <a:tab/> */
        const char *q = t + 4;
        if (*q != '>' && !isspace((unsigned char)*q)) { p = t + 4; continue; }
        const char *gt = memchr(q, '>', end - q);
        if (!gt) break;
        const char *close = strstr(gt, "</a:t>");
        if (!close || close > end) break;
        /* paragraph break marker: if a </a:p> occurred BEFORE this run since
         * the previous run, emit '\n' first (paragraphs separate lines) */
        for (const char *s2 = p; s2 < gt; s2++) {
            if (s2 + 6 <= end && !strncmp(s2, "</a:p>", 6)) {
                if (n + 1 >= cap) { cap *= 2; char *t2 = realloc(out, cap); if (!t2) { free(out); return NULL; } out = t2; }
                out[n++] = '\n'; out[n] = 0;
                break;
            }
        }
        for (const char *s = gt + 1; s < close; s++) {
            if (n + 8 >= cap) { cap *= 2; char *t2 = realloc(out, cap); if (!t2) { free(out); return NULL; } out = t2; }
            out[n++] = *s;
        }
        out[n] = 0;
        p = close + 6;
    }
    /* trailing newline if document ends inside/after a closed paragraph */
    {
        const char *pe = NULL;
        for (const char *s2 = p; s2 + 6 <= end; s2++)
            if (!strncmp(s2, "</a:p>", 6)) { pe = s2; break; }
        if (pe && n && out[n-1] != '\n') {
            if (n + 1 >= cap) { cap *= 2; char *t2 = realloc(out, cap); if (!t2) { free(out); return NULL; } out = t2; }
            out[n++] = '\n'; out[n] = 0;
        }
    }
    xml_unescape(out);
    return out;
}

/* Run `cmd`, read all stdout into heap buffer. Returns NULL on failure. */
static char *slurp_cmd(const char *cmd, size_t *len_out) {
    FILE *f = popen(cmd, "r");
    if (!f) return NULL;
    size_t cap = 4096, n = 0;
    char *buf = malloc(cap);
    if (!buf) { pclose(f); return NULL; }
    size_t r;
    while ((r = fread(buf + n, 1, cap - n, f)) > 0) {
        n += r;
        if (n == cap) { cap *= 2; char *t = realloc(buf, cap); if (!t) { free(buf); pclose(f); return NULL; } buf = t; }
    }
    pclose(f);
    buf[n] = 0;
    if (len_out) *len_out = n;
    return buf;
}

int cp_script_load_pptx(cp_script *s, const char *path, char *err, size_t errn) {
    /* sanity: zip magic PK\x03\x04 */
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errn, "cannot open %s", path); return -1; }
    unsigned char m[4];
    if (fread(m, 1, 4, f) != 4 || m[0] != 'P' || m[1] != 'K') {
        fclose(f); snprintf(err, errn, "not a .pptx (no ZIP header)"); return -1;
    }
    fclose(f);

    /* list notes slides present */
    char cmd[1024 + 32];
    snprintf(cmd, sizeof cmd, "unzip -Z1 '%s' 'ppt/notesSlides/notesSlide*.xml' 2>/dev/null", path);
    char *listing = slurp_cmd(cmd, NULL);
    if (!listing || !*listing) {
        free(listing);
        snprintf(err, errn, "no presenter notes found in %s", path);
        return -1;
    }

    note_t notes[MAXPAGES]; int nn = 0;
    memset(notes, 0, sizeof notes);
    for (char *line = strtok(listing, "\n"); line && nn < MAXPAGES; line = strtok(NULL, "\n")) {
        int num = 0;
        const char *base = strrchr(line, '/'); base = base ? base + 1 : line;
        if (sscanf(base, "notesSlide%d.xml", &num) != 1) continue;
        /* extract this entry */
        char cmd2[1024 + 512];
        snprintf(cmd2, sizeof cmd2, "unzip -p '%s' '%s' 2>/dev/null", path, line);
        size_t xlen = 0;
        char *xml = slurp_cmd(cmd2, &xlen);
        if (!xml) continue;
        char *txt = cp_pptx_extract_at_texts(xml, xlen);
        free(xml);
        if (!txt) continue;
        /* trim leading/trailing blank lines */
        char *a = txt; while (*a == '\n' || *a == ' ' || *a == '\t') a++;
        size_t tl = strlen(a);
        while (tl && (a[tl-1] == '\n' || a[tl-1] == ' ' || a[tl-1] == '\t')) a[--tl] = 0;
        if (tl) { notes[nn].n = num; notes[nn].text = a; nn++; }
        else free(txt);
    }
    free(listing);

    if (nn == 0) { snprintf(err, errn, "notes slides empty"); return -1; }
    qsort(notes, nn, sizeof notes[0], cmp_note);

    /* join into one big text with "# " page headings so load_text splits pages */
    size_t total = 0;
    for (int i = 0; i < nn; i++) total += strlen(notes[i].text) + 32;
    char *big = malloc(total + 1);
    if (!big) { for (int i = 0; i < nn; i++) free(notes[i].text); return -1; }
    big[0] = 0;
    for (int i = 0; i < nn; i++) {
        strcat(big, "# Slide ");
        char tmp[16]; snprintf(tmp, sizeof tmp, "%d\n", notes[i].n); strcat(big, tmp);
        strcat(big, notes[i].text);
        strcat(big, "\n\n");
        free(notes[i].text);
    }
    int rc = cp_script_load_text(s, big, 1);
    free(big);
    return rc;
}
