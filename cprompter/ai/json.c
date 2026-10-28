/* ai/json.c — see json.h */
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static const char *skip_ws(const char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    return s;
}

/* parse a JSON string starting at the opening quote into out; returns end
 * (just past closing quote) or NULL on error. Handles \", \\, \/, \b, \f,
 * \n, \r, \t, \uXXXX (BMP only: UTF-8 encoded; surrogate pairs passed
 * through as replacement chars — acceptable for coach text). */
static const char *parse_string(const char *s, char *out, size_t outn) {
    if (*s != '"') return NULL;
    s++;
    size_t o = 0;
    while (*s) {
        char c = *s;
        if (c == '"') { if (out && o < outn) out[o] = 0; return s + 1; }
        if (c == '\\') {
            s++;
            char e = *s;
            char r = 0;
            switch (e) {
                case '"': r='"'; break; case '\\': r='\\'; break;
                case '/': r='/'; break; case 'b': r='\b'; break;
                case 'f': r='\f'; break; case 'n': r='\n'; break;
                case 'r': r='\r'; break; case 't': r='\t'; break;
                case 'u': {
                    unsigned cp = 0;
                    for (int i = 1; i <= 4; i++) {
                        char h = s[i];
                        if (!isxdigit((unsigned char)h)) return NULL;
                        cp = cp * 16 + (isdigit((unsigned char)h) ? h - '0'
                             : (tolower(h) - 'a' + 10));
                    }
                    s += 4;
                    /* encode BMP codepoint as UTF-8 */
                    if (cp >= 0xD800 && cp <= 0xDFFF) {
                        if (out && o + 3 < outn) { memcpy(out+o, "\xEF\xBF\xBD", 3); o += 3; }
                    } else if (cp < 0x80) {
                        if (out && o + 1 < outn) out[o++] = (char)cp;
                    } else if (cp < 0x800) {
                        if (out && o + 2 < outn) {
                            out[o++] = (char)(0xC0 | (cp >> 6));
                            out[o++] = (char)(0x80 | (cp & 0x3F));
                        }
                    } else {
                        if (out && o + 3 < outn) {
                            out[o++] = (char)(0xE0 | (cp >> 12));
                            out[o++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                            out[o++] = (char)(0x80 | (cp & 0x3F));
                        }
                    }
                    s++;
                    continue;
                }
                default: return NULL;
            }
            if (out && o + 1 < outn) out[o++] = r;
            s++;
            continue;
        }
        if (out && o + 1 < outn) out[o++] = c;
        s++;
    }
    return NULL;
}

/* skip any JSON value, returning pointer just past it */
static const char *skip_value(const char *s) {
    s = skip_ws(s);
    if (*s == '"') return parse_string(s, NULL, 0);
    if (*s == '{' || *s == '[') {
        int depth = 0;
        for (; *s; s++) {
            if (*s == '"' ) { s = parse_string(s, NULL, 0); if (!s) return NULL; s--; continue; }
            if (*s == '{' || *s == '[') depth++;
            else if (*s == '}' || *s == ']') { depth--; if (depth == 0) return s + 1; }
        }
        return NULL;
    }
    while (*s && *s != ',' && *s != '}' && *s != ']' && !isspace((unsigned char)*s)) s++;
    return s;
}

/* find ':'-terminated key occurrence at object-member position */
static const char *find_key(const char *json, const char *key) {
    size_t klen = strlen(key);
    const char *p = json;
    while ((p = strchr(p, '"')) != NULL) {
        const char *kend = parse_string(p, NULL, 0);
        if (!kend) return NULL;
        /* is this key followed by colon and matching? */
        const char *after = skip_ws(kend);
        if (*after == ':') {
            /* compare raw (no escapes in keys we use) */
            size_t len = (size_t)(kend - p - 2);
            if (len == klen && !strncmp(p + 1, key, klen))
                return skip_ws(after + 1);
        }
        p = kend;
    }
    return NULL;
}

int cp_json_get_string(const char *json, const char *key, char *out, size_t outn) {
    const char *v = find_key(json, key);
    if (!v || *v != '"') return 0;
    return parse_string(v, out, outn) ? 1 : 0;
}

/* advance to the nth element of an array starting at '[' */
static const char *array_elem(const char *s, size_t idx) {
    s = skip_ws(s);
    if (*s != '[') return NULL;
    s++;
    size_t i = 0;
    while (*s) {
        s = skip_ws(s);
        if (*s == ']') return NULL;
        const char *ve = skip_value(s);
        if (!ve) return NULL;
        if (i == idx) return s;
        i++;
        s = skip_ws(ve);
        if (*s == ',') s++;
    }
    return NULL;
}

int cp_json_find_path(const char *json, const char *path, char *out, size_t outn) {
    const char *cur = json;
    char seg[128];
    const char *pp = path;
    while (*pp) {
        const char *dot = strchr(pp, '.');
        size_t sl = dot ? (size_t)(dot - pp) : strlen(pp);
        if (sl >= sizeof seg) return 0;
        memcpy(seg, pp, sl); seg[sl] = 0;
        pp = dot ? dot + 1 : pp + strlen(pp);
        cur = skip_ws(cur);
        if (isdigit((unsigned char)seg[0]) && strchr("0123456789", seg[0]) && sl > 0) {
            /* numeric segment: could index an array OR be an object key like
             * {"0":...}. Try array first when current is '['. */
            if (*cur == '[') { cur = array_elem(cur, strtoul(seg, NULL, 10)); if (!cur) return 0; continue; }
        }
        const char *v = find_key(cur, seg);
        if (!v) return 0;
        cur = v;
    }
    if (*cur == '"') return parse_string(cur, out, outn) ? 1 : 0;
    return 0;
}

double cp_json_get_number(const char *json, const char *path, double def) {
    /* reuse walker: navigate to parent then find last segment as number */
    const char *cur = json;
    char seg[128];
    const char *pp = path;
    const char *lastval = NULL;
    while (*pp) {
        const char *dot = strchr(pp, '.');
        size_t sl = dot ? (size_t)(dot - pp) : strlen(pp);
        if (sl >= sizeof seg) return def;
        memcpy(seg, pp, sl); seg[sl] = 0;
        pp = dot ? dot + 1 : pp + strlen(pp);
        cur = skip_ws(cur);
        if (*cur == '[' && strspn(seg, "0123456789") == sl && sl > 0) {
            cur = array_elem(cur, strtoul(seg, NULL, 10)); if (!cur) return def;
            lastval = cur;
            continue;
        }
        const char *v = find_key(cur, seg);
        if (!v) return def;
        cur = v; lastval = cur;
    }
    if (!lastval) return def;
    if (!(isdigit((unsigned char)*lastval) || *lastval=='-' || *lastval=='+')) return def;
    return strtod(lastval, NULL);
}
