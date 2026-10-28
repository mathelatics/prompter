/* script_io.c — tiny JSON reader/writer for .textream files.
 * Reader strategy: we only need a handful of known keys, so instead of a full
 * DOM we run a string-aware key scanner that finds `"key" : value` pairs at
 * any depth and extracts strings/numbers. Pages array is handled by splitting
 * on top-level object boundaries inside "pages":[ ... ]. */
#include "script_io.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- writer -------------------------------------------------------- */

static void json_escape_str(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '"':  fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\n': fputs("\\n",  f); break;
        case '\r': fputs("\\r",  f); break;
        case '\t': fputs("\\t",  f); break;
        default:
            if (*p < 0x20) fprintf(f, "\\u%04x", *p);
            else fputc(*p, f);
        }
    }
    fputc('"', f);
}

static const char *mode_name(cp_mode m) {
    switch (m) {
    case CP_MODE_TRACK: return "wordTracking";
    case CP_MODE_VOICE: return "voiceActivated";
    default:            return "classic";
    }
}
static const char *color_names[6] = {"white","yellow","green","blue","pink","orange"};
static const char *font_names[4]  = {"sans","serif","mono","dyslexic"};

int cp_script_save_textream(const cp_script *s, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fputs("{\n  \"version\": \"1.0\",\n  \"pages\": [\n", f);
    for (size_t p = 0; p < s->np; p++) {
        const cp_page *pg = &s->pages[p];
        fprintf(f, "    { \"id\": %zu, \"name\": ", p + 1);
        json_escape_str(f, pg->name);
        fputs(", \"content\": ", f);
        /* rebuild page text from words (single spaces; paragraphs as \n\n) */
        fputc('"', f);
        int lastpara = -1;
        for (int i = pg->w0; i < pg->w1; i++) {
            if (lastpara != -1) {
                if (s->words[i].para != lastpara) fputs("\\n\\n", f);
                else fputc(' ', f);
            }
            for (const unsigned char *q = (const unsigned char *)s->words[i].text; *q; q++) {
                if (*q == '"' || *q == '\\') { fputc('\\', f); fputc(*q, f); }
                else if (*q == '\n') fputs("\\n", f);
                else fputc(*q, f);
            }
            lastpara = s->words[i].para;
        }
        fputc('"', f);
        cp_mode m = pg->mode != (cp_mode)-1 ? pg->mode : s->default_mode;
        double sp = pg->scroll_speed > 0 ? pg->scroll_speed : s->default_speed;
        fprintf(f, ",\n      \"settings\": { \"guidanceMode\": \"%s\","
                  " \"scrollSpeed\": %.2f, \"speechLanguage\": \"%s\" } }%s\n",
                mode_name(m), sp, s->lang,
                p + 1 < s->np ? "," : "");
    }
    fputs("  ],\n  \"settings\": {", f);
    fprintf(f, " \"mirror\": %d,", s->mirror);
    fprintf(f, " \"fontFamily\": \"%s\",", font_names[s->font_family & 3]);
    fprintf(f, " \"fontSize\": %d,", s->font_size);
    fprintf(f, " \"highlightColor\": \"%s\"", color_names[s->highlight_color % 6]);
    fputs(" },\n  \"metadata\": { \"generator\": \"cprompter\" }\n}\n", f);
    fclose(f);
    return 0;
}

int cp_script_save_txt(const cp_script *s, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    for (size_t p = 0; p < s->np; p++) {
        fprintf(f, "# %s\n", s->pages[p].name);
        int lastpara = -1;
        for (int i = s->pages[p].w0; i < s->pages[p].w1; i++) {
            if (lastpara != -1)
                fputs(s->words[i].para != lastpara ? "\n\n" : " ", f);
            fputs(s->words[i].text, f);
            lastpara = s->words[i].para;
        }
        fputc('\n', f);
    }
    fclose(f);
    return 0;
}

/* ---------- reader (minimal key scanner) ---------------------------------- */

typedef struct { const char *p; const char *end; } J;

static void skipws(J *j) {
    while (j->p < j->end && isspace((unsigned char)*j->p)) j->p++;
}
/* find next occurrence of "needle" outside of strings; returns NULL if none */
/* parse a JSON string starting at *pp (which points at opening quote);
 * unescapes into out (heap). advances *pp past closing quote. */
static char *parse_string(J *j) {
    skipws(j);
    if (j->p >= j->end || *j->p != '"') return NULL;
    j->p++;
    size_t cap = 64, n = 0;
    char *out = malloc(cap);
    if (!out) return NULL;
    while (j->p < j->end && *j->p != '"') {
        char c = *j->p++;
        if (c == '\\' && j->p < j->end) {
            char e = *j->p++;
            switch (e) {
            case 'n': c = '\n'; break;
            case 'r': c = '\r'; break;
            case 't': c = '\t'; break;
            case '"': c = '"';  break;
            case '\\': c = '\\'; break;
            case '/': c = '/';  break;
            case 'u': {
                /* decode BMP \uXXXX to UTF-8 (skip surrogates for brevity) */
                if (j->end - j->p >= 4) {
                    char hx[5] = {j->p[0],j->p[1],j->p[2],j->p[3],0};
                    unsigned cpv = (unsigned)strtoul(hx, NULL, 16);
                    j->p += 4;
                    if (cpv < 0x80) c = (char)cpv;
                    else if (cpv < 0x800) { /* two-byte */
                        if (n + 2 >= cap) { cap *= 2; out = realloc(out, cap); }
                        out[n++] = (char)(0xC0 | (cpv >> 6));
                        c = (char)(0x80 | (cpv & 0x3F));
                    } else {
                        if (n + 2 >= cap) { cap *= 2; out = realloc(out, cap); }
                        out[n++] = (char)(0xE0 | (cpv >> 12));
                        out[n++] = (char)(0x80 | ((cpv >> 6) & 0x3F));
                        c = (char)(0x80 | (cpv & 0x3F));
                    }
                }
                break;
            }
            default: c = e;
            }
        }
        if (n + 1 >= cap) { cap *= 2; out = realloc(out, cap); }
        out[n++] = c;
    }
    if (j->p < j->end && *j->p == '"') j->p++;
    out[n] = 0;
    return out;
}
static double parse_number(J *j, double dflt) {
    skipws(j);
    char *e;
    double v = strtod(j->p, &e);
    if (e != j->p) { j->p = e; return v; }
    return dflt;
}

/* helper: set j->p just after `"key":` and read a value of expected kind.
 * NOTE: find_outside_strings scans forward from j->p but the needle itself is
 * a quoted token; re-scan must start at base.p (the state machine there only
 * tracks strings, so a needle containing quotes still matches correctly). */
static int get_key_value(J base, const char *key, J *val_out) {
    J j = base;
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    size_t nl = strlen(pat);
    const char *hit = NULL;
    /* string-aware scan: skip over JSON string values so a key name that
     * appears inside script text ("content") never produces a false match.
     * The quoted-key token is recognized when '"' starts at a non-instr
     * position and the following chars equal the key plus closing quote and
     * optional whitespace then ':'. */
    int instr = 0;
    for (const char *p = j.p; p < j.end; ) {
        if (instr) {
            if (*p == '\\') { p += 2; continue; }
            if (*p == '"') instr = 0;
            p++;
            continue;
        }
        if (*p == '"') {
            /* possible key occurrence starting here */
            if ((size_t)(j.end - p) >= nl && memcmp(p, pat, nl) == 0) {
                const char *q = p + nl;
                while (q < j.end && isspace((unsigned char)*q)) q++;
                if (q < j.end && *q == ':') { hit = p; break; }
            }
            instr = 1; p++;
            continue;
        }
        p++;
    }
    if (!hit) return 0;
    j.p = hit + nl;
    skipws(&j);
    if (j.p < j.end && *j.p == ':') { j.p++; skipws(&j); *val_out = j; return 1; }
    return 0;
}

static cp_mode mode_from_str(const char *m, cp_mode dflt) {
    if (!m) return dflt;
    if (!strcmp(m, "wordTracking"))  return CP_MODE_TRACK;
    if (!strcmp(m, "voiceActivated")) return CP_MODE_VOICE;
    if (!strcmp(m, "classic"))        return CP_MODE_CLASSIC;
    return dflt;
}

int cp_script_load_textream(cp_script *s, const char *path, char *err, size_t errn) {
    if (errn) err[0] = 0;
    FILE *f = fopen(path, "rb");
    if (!f) { if (errn) snprintf(err, errn, "cannot open %s", path); return -1; }
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    if (len <= 0 || len > 64 * 1024 * 1024) {
        fclose(f);
        if (errn) snprintf(err, errn, "bad size");
        return -1;
    }
    char *buf = malloc(len + 1);
    if (!buf || fread(buf, 1, len, f) != (size_t)len) {
        free(buf); fclose(f);
        if (errn) snprintf(err, errn, "read error");
        return -1;
    }
    buf[len] = 0; fclose(f);

    J doc = { buf, buf + len };
    int rc = 0;

    /* top-level settings first. "settings" is an OBJECT; we must scan only
     * inside it, otherwise page-level keys (speechLanguage etc.) leak up and
     * the value cursor lands on the object's '{' instead of a real value. */
    J v;
    J sset = doc;   /* fallback: whole doc (flat files) */
    if (get_key_value(doc, "settings", &v)) sset = v;

    if (get_key_value(sset, "speechLanguage", &v)) {
        J t = v; char *lang = parse_string(&t);
        if (lang) { snprintf(s->lang, sizeof s->lang, "%s", lang); free(lang); }
    }
    if (get_key_value(sset, "fontSize", &v))       s->font_size = (int)parse_number(&v, s->font_size);
    if (get_key_value(sset, "mirror", &v))         s->mirror = (int)parse_number(&v, 0);
    if (get_key_value(sset, "fontFamily", &v)) {
        char *ff = parse_string(&v);
        if (ff) { s->font_family = !strcmp(ff,"serif")?1:!strcmp(ff,"mono")?2:!strcmp(ff,"dyslexic")?3:0; free(ff); }
    }
    if (get_key_value(sset, "highlightColor", &v)) {
        char *hc = parse_string(&v);
        if (hc) { for (int i = 0; i < 6; i++) if (!strcmp(hc, color_names[i])) s->highlight_color = i; free(hc); }
    }

    /* pages array: iterate objects between '{' and matching '}' (no nested
     * objects inside our page schema except "settings", which we scan by key) */
    if (!get_key_value(doc, "pages", &v)) {
        /* fall back: treat whole file content key "content" or raw text */
        if (errn) snprintf(err, errn, "no pages[]; not a .textream file");
        rc = -2; goto done;
    }
    {
        /* walk the array, extracting each element substring */
        J arr = v;
        if (arr.p >= arr.end || *arr.p != '[') { rc = -1; goto done; }
        arr.p++;
        /* collect page contents/modes/speeds into temp arrays */
        size_t pc = 0, pcap = 8;
        typedef struct { char *content; char *name; cp_mode mode; double speed; char *lang; } P;
        P *ps = malloc(pcap * sizeof *ps);
        while (arr.p < arr.end) {
            skipws(&arr);
            if (arr.p >= arr.end) break;
            if (*arr.p == ']') { arr.p++; break; }
            if (*arr.p != '{') { arr.p++; continue; }
            /* find matching close brace (string-aware) */
            const char *st = arr.p; int depth = 0, instr = 0;
            const char *q = st;
            for (; q < arr.end; q++) {
                if (instr) { if (*q == '\\') q++; else if (*q == '"') instr = 0; continue; }
                if (*q == '"') { instr = 1; continue; }
                if (*q == '{') depth++;
                else if (*q == '}') { depth--; if (!depth) { q++; break; } }
            }
            J obj = { st, q };
            if (pc == pcap) { pcap *= 2; ps = realloc(ps, pcap * sizeof *ps); }
            P *p = &ps[pc++];
            memset(p, 0, sizeof *p);
            p->mode = (cp_mode)-1;
            J t;
            if (get_key_value(obj, "content", &t)) p->content = parse_string(&t);
            if (get_key_value(obj, "name", &t))    p->name = parse_string(&t);
            /* guidanceMode/scrollSpeed/speechLanguage live inside the page's
             * "settings" object — scope the scan there so we never match a
             * substring inside "content" text */
            J sobj;  /* page settings sub-object */
            if (get_key_value(obj, "settings", &sobj)) {
                if (get_key_value(sobj, "guidanceMode", &t)) {
                    char *m = parse_string(&t);
                    p->mode = mode_from_str(m, (cp_mode)-1);
                    free(m);
                }
                if (get_key_value(sobj, "scrollSpeed", &t)) p->speed = parse_number(&t, 0);
                if (get_key_value(sobj, "speechLanguage", &t)) p->lang = parse_string(&t);
            }
            arr.p = q;
        }
        /* stitch pages: load each content via tokenizer, then override cfg */
        /* We build one big text with "# name" separators so tokenizer splits */
        size_t need = 2;
        for (size_t i = 0; i < pc; i++) need += (ps[i].content ? strlen(ps[i].content) : 0)
                                             + (ps[i].name ? strlen(ps[i].name) : 8) + 8;
        char *big = malloc(need);
        if (!big) { free(ps); rc = -1; goto done; }
        big[0] = 0;
        size_t off = 0;
        for (size_t i = 0; i < pc; i++) {
            off += snprintf(big + off, need - off, "# %s\n%s\n\n",
                            ps[i].name ? ps[i].name : "Page",
                            ps[i].content ? ps[i].content : "");
        }
        cp_script_load_text(s, big, 1);
        free(big);
        /* apply per-page settings now that pages exist in order */
        for (size_t i = 0; i < pc && i < s->np; i++) {
            if (ps[i].mode != (cp_mode)-1) s->pages[i].mode = ps[i].mode;
            if (ps[i].speed > 0)           s->pages[i].scroll_speed = ps[i].speed;
            if (ps[i].lang)                snprintf(s->lang, sizeof s->lang, "%s", ps[i].lang);
        }
        if (pc) {
            /* document-level default mode = first page's mode if set */
            if (s->pages[0].mode != (cp_mode)-1) s->default_mode = s->pages[0].mode;
        }
        for (size_t i = 0; i < pc; i++) { free(ps[i].content); free(ps[i].name); free(ps[i].lang); }
        free(ps);
    }

done:
    free(buf);
    return rc;
}
