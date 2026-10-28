/* pacer.c — guidance-mode state machine. See plan §2: three modes feed one
 * fractional word position. Manual scroll pauses the clock and resumes from
 * the new position (Textream behavior). max_delivered is monotone -> drives
 * director read-lock (§4.5) and remote progress. */
#include "pacer.h"

#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static cp_mode active_mode(const cp_pacer *p) {
    size_t i = cp_pacer_word_index(p);
    int pg = cp_page_of(p->s, i);
    if (pg >= 0 && p->s->pages[pg].mode != (cp_mode)-1)
        return p->s->pages[pg].mode;
    return p->s->default_mode;
}

static double active_speed(const cp_pacer *p) {
    size_t i = cp_pacer_word_index(p);
    int pg = cp_page_of(p->s, i);
    if (pg >= 0 && p->s->pages[pg].scroll_speed > 0)
        return p->s->pages[pg].scroll_speed;
    return p->speed_wps > 0 ? p->speed_wps : p->s->default_speed;
}

static void clamp(cp_pacer *p) {
    double hi = (double)p->s->nw;
    if (p->pos < 0) p->pos = 0;
    if (p->pos > hi) p->pos = hi;
    if (p->pos > p->max_delivered) p->max_delivered = p->pos;
}

void cp_pacer_init(cp_pacer *p, const cp_script *s, double start_speed_wps) {
    memset(p, 0, sizeof *p);
    p->s = s;
    p->speed_wps = start_speed_wps > 0 ? start_speed_wps : s->default_speed;
    p->measured_wpm = 0;
    p->track_target = 0;
}

void cp_pacer_tick(cp_pacer *p, double now_sec, double dt) {
    if (dt <= 0) return;
    if (p->paused) { p->t_last = now_sec; return; }

    switch (active_mode(p)) {
    case CP_MODE_CLASSIC:
        p->pos += active_speed(p) * dt;
        break;
    case CP_MODE_VOICE:
        /* Textream: move only while speaking; rate = max(min speed, observed) */
        if (p->vad_active) {
            double sp = active_speed(p);
            double obs = p->measured_wpm / 60.0;
            p->pos += (obs > sp ? obs : sp) * dt;
        }
        break;
    case CP_MODE_TRACK:
        /* ease toward ASR-aligned target (critically-damped-ish lerp).
         * Never jump backward automatically (plan §4.1). */
        if (p->track_target > p->pos) {
            double gap = p->track_target - p->pos;
            double follow = gap * (1.0 - exp(-3.0 * dt));   /* ~0.95/s ease */
            if (follow > active_speed(p) * 2.5 * dt)        /* cap catch-up */
                follow = active_speed(p) * 2.5 * dt;
            p->pos += follow;
        }
        p->drifted_by = (int)(p->track_target - p->pos);
        break;
    }
    clamp(p);
    p->t_last = now_sec;
}

void cp_pacer_scroll_by(cp_pacer *p, double delta_words) {
    /* mouse-wheel override: repositions and keeps running from there */
    p->pos += delta_words;
    clamp(p);
}

void cp_pacer_jump(cp_pacer *p, double word_idx, int allow_back) {
    /* explicit jumps (tap-to-jump, keys): backward allowed only when asked */
    if (!allow_back && word_idx < p->pos) return;
    p->pos = word_idx;
    if (active_mode(p) == CP_MODE_TRACK) p->track_target = word_idx;
    clamp(p);
}

void cp_pacer_set_paused(cp_pacer *p, int paused) { p->paused = paused; }

void cp_pacer_feed_vad(cp_pacer *p, int active, double inst_wpm) {
    p->vad_active = active;
    if (active && inst_wpm > 0)
        p->measured_wpm = p->measured_wpm <= 0
                        ? inst_wpm
                        : p->measured_wpm * 0.8 + inst_wpm * 0.2;  /* EMA */
}

void cp_pacer_feed_align(cp_pacer *p, double aligned_pos) {
    /* Word-tracking is authoritative: unlike VAD mode we DO allow small
     * backward moves (stutters/re-reads align.c resolves) but never below
     * floor(max_delivered)-AL_BACK_CAP so we don't rewind whole paragraphs.
     * The current track_target itself counts as "recently shown", which is
     * what lets a stutter re-read the last few words without rewinding the
     * whole paragraph (align.c emits raw cursor positions; M2 audit §4.1). */
    if (aligned_pos >= p->track_target) {          /* forward: always ok   */
        p->track_target = aligned_pos;
    } else {
        /* Bounded backtrack: re-reads may pull the cursor back, but never
         * below max_delivered-6 NOR below the display position itself —
         * a target behind pos would make tick() stall/reverse the render.
         * (matches align.c AL_MAX_BACK semantics) */
        double lo = p->max_delivered - 6.0;
        if (lo < p->pos - 1.0) lo = p->pos - 1.0;  /* never stall the ease */
        if (aligned_pos > lo && !p->paused) p->track_target = aligned_pos;
    }
}

void cp_pacer_resync(cp_pacer *p) {
    /* "tap to resync" (hotkey): snap pos onto the tracker cursor at once. */
    if (p->track_target > p->pos) p->pos = p->track_target;
    clamp(p);
}

size_t cp_pacer_word_index(const cp_pacer *p) {
    long i = (long)floor(p->pos);
    if (i < 0) i = 0;
    if ((size_t)i >= p->s->nw) i = (long)(p->s->nw ? p->s->nw - 1 : 0);
    return (size_t)i;
}

/* ---- M3: director read-lock ---------------------------------------------
 * max_delivered is the monotone high-water mark of words shown to the
 * audience. Textream's Director Mode refuses edits to text already read;
 * we enforce that on the TOKEN stream (authoritative, tokenizer-exact) by
 * serializing delivered words + raw tail into a canonical snapshot string.
 * An incoming edit must keep that snapshot as an exact prefix; if it does,
 * we splice new_text onto the unchanged tail and re-tokenize in place. */

/* Serialize the first `n` script words followed by the untouched remainder
 * of raw starting at word n's byte offset. Words are joined with single
 * spaces so any equivalent whitespace spelling collapses to one canon. */
static char *serialize_upto(const cp_script *s, size_t n) {

    if (!s || !s->raw) return NULL;
    size_t cap = 0;
    for (size_t i = 0; i < n && i < s->nw; i++) cap += s->words[i].n + 1;
    cap += strlen(s->raw) + 2;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    size_t k = 0;
    for (size_t i = 0; i < n && i < s->nw; i++) {
        if (i) buf[k++] = ' ';
        memcpy(buf + k, s->words[i].text, s->words[i].n);
        k += s->words[i].n;
    }
    /* append raw from the end-offset of the last serialized word */
    size_t off = 0;
    if (n > 0 && n <= s->nw) {
        const char *last = s->words[n - 1].text;
        off = (size_t)(last - s->raw) + s->words[n - 1].n;
    }
    size_t rl = strlen(s->raw + off);
    memcpy(buf + k, s->raw + off, rl + 1);
    return buf;
}

size_t cp_script_chars_for_words(const cp_script *s, double words) {
    if (!s || !s->raw || s->raw[0] == 0) return 0;
    size_t n = (size_t)floor(words);
    if (n >= s->nw) return strlen(s->raw);
    if (n == 0) return 0;
    const cp_word *w = &s->words[n - 1];
    return (size_t)(w->text - s->raw) + w->n;   /* end offset of nth word */
}

/* ---- canonical word-prefix compare -------------------------------------
 * The read-lock must survive benign reformatting (director web UI collapses
 * newlines into spaces, editors re-wrap paragraphs) while still catching a
 * real rewrite of delivered text. So we compare WORD streams, not bytes:
 * each word is lowercased with leading/trailing ASCII punctuation stripped
 * (UTF-8 continuation bytes are >=0x80 so they never get mangled). Two
 * strings match iff their first `n` canonical words are equal.
 * Returns 1 if the first n canonical words of a and b agree (or a has fewer
 * than n words but then b must have exactly the same count), else 0. */
static int canon_word_eq(const char **pa, const char **pb) {
    /* advance each pointer over one canonical word; compare contents */
    const char *a = *pa, *b = *pb;
    while (*a && (unsigned char)*a <= ' ') a++;
    while (*b && (unsigned char)*b <= ' ') b++;
    const char *ae = a, *be = b;
    while (*ae && (unsigned char)*ae > ' ') ae++;
    while (*be && (unsigned char)*be > ' ') be++;
    /* strip trailing punctuation on both sides */
    while (ae > a && ispunct((unsigned char)ae[-1])) ae--;
    while (be > b && ispunct((unsigned char)be[-1])) be--;
    /* strip leading punctuation */
    while (a < ae && ispunct((unsigned char)*a)) a++;
    while (b < be && ispunct((unsigned char)*b)) b++;
    size_t an = (size_t)(ae - a), bn = (size_t)(be - b);
    if (an != bn) return 0;
    for (size_t i = 0; i < an; i++)
        if (tolower((unsigned char)a[i]) != tolower((unsigned char)b[i])) return 0;
    *pa = ae; *pb = be;
    return 1;
}

static int canon_prefix_match(const char *a, const char *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!canon_word_eq(&a, &b)) return 0;      /* mismatch or a exhausted */
    }
    return 1;
}

int cp_pacer_apply_edit(cp_pacer *p, const char *new_text, char *err, size_t errn) {
    if (err && errn) err[0] = 0;
    if (!p || !p->s || !new_text) {
        if (err) snprintf(err, errn, "invalid arguments");
        return -1;
    }
    cp_script *m = (cp_script *)p->s;
    /* Fallback for scripts without raw text (.textream loads words only):
     * tokenize the candidate first and compare canonical WORD streams, which
     * is exactly the read-lock semantics we want anyway. */
    if (!m->raw) {
        cp_script *tmp = cp_script_new();
        if (cp_script_load_text(tmp, new_text, m->raw_split_pages) != 0) {
            if (err) snprintf(err, errn, "cannot tokenize edit");
            cp_script_free(tmp);
            return -1;
        }
        size_t nlock = (size_t)floor(p->max_delivered);
        if (nlock > m->nw) nlock = m->nw;
        int conflict = 0;
        if (nlock > tmp->nw) conflict = 1;         /* edit deleted delivered words */
        for (size_t i = 0; i < nlock && !conflict; i++) {
            const char *a = tmp->words[i].text, *b = m->words[i].text;
            size_t alen = tmp->words[i].n, blen = m->words[i].n;
            while (alen && ispunct((unsigned char)a[alen-1])) alen--;
            while (blen && ispunct((unsigned char)b[blen-1])) blen--;
            size_t ao = 0, bo = 0;
            while (ao < alen && ispunct((unsigned char)a[ao])) ao++;
            while (bo < blen && ispunct((unsigned char)b[bo])) bo++;
            size_t an = alen - ao, bn = blen - bo;   /* canonical lengths */
            if (an != bn) { conflict = 1; break; }
            for (size_t k = 0; k < an; k++)
                if (tolower((unsigned char)a[ao+k]) != tolower((unsigned char)b[bo+k]))
                    { conflict = 1; break; }
        }
        if (conflict) {
            if (err) snprintf(err, errn,
                "read-lock: first %zu delivered word(s) must not change", nlock);
            cp_script_free(tmp);
            return -1;
        }
        /* adopt tokenized content: copy raw back into m via load path */
        cp_script_load_text(m, new_text, m->raw_split_pages);
        cp_script_free(tmp);
        if (p->max_delivered > (double)m->nw) p->max_delivered = m->nw;
        if (p->pos > (double)m->nw) p->pos = m->nw;
        if (p->track_target > (double)m->nw) p->track_target = m->nw;
        return 0;
    }
    size_t nlock = (size_t)floor(p->max_delivered);
    if (nlock > m->nw) nlock = m->nw;
    char *locked = serialize_upto(m, nlock);
    if (!locked) {
        if (err) snprintf(err, errn, "out of memory");
        return -1;
    }
    /* canonical word-prefix compare: whitespace/case/punctuation reformatting
     * is allowed, changing delivered WORDS is not */
    int conflict = !canon_prefix_match(new_text, locked, nlock);
    free(locked);
    if (conflict) {
        if (err) snprintf(err, errn,
            "read-lock: first %zu delivered word(s) must not change", nlock);
        return -1;
    }
    /* safe: re-tokenize in place (same cp_script* keeps identity) */
    cp_script_load_text(m, new_text, m->raw_split_pages);
    /* clamp pacer state into the new bounds; keep high-water semantics */
    if (p->max_delivered > (double)m->nw) p->max_delivered = m->nw;
    if (p->pos > (double)m->nw) p->pos = m->nw;
    if (p->track_target > (double)m->nw) p->track_target = m->nw;
    return 0;
}

int cp_pacer_finished(const cp_pacer *p) {
    return p->s->nw > 0 && p->pos >= (double)p->s->nw - 0.001;
}

double cp_pacer_remaining_seconds(const cp_pacer *p) {
    double rem = (double)p->s->nw - p->pos;
    double sp = active_speed(p);
    if (sp <= 0) sp = 1e-9;
    return rem > 0 ? rem / sp : 0;
}
