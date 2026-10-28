/* ai/coach.c — see coach.h. Rules engine + throttled LLM worker thread. */
#define _POSIX_C_SOURCE 200809L
#include "coach.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

struct cp_coach {
    cp_coach_params p;
    pthread_mutex_t mu;
    cp_advice mailbox;      /* newest advice (any source)   */
    int       mbox_seq;
    /* rule bookkeeping */
    double last_rule_slow_at, last_rule_fast_at, last_rule_silence_at;
    double last_rule_drift_at;
    double t_silent_since;  /* 0 = speaking/paused          */
    double last_llm_at;
    int    llm_busy;
    int    running;         /* worker thread lifecycle      */
    pthread_t th;
    /* snapshot the worker reads without touching UI state races: */
    double snap_pos, snap_maxdel, snap_speed, snap_wpm;
    size_t snap_nw; int snap_page, snap_word;
    char   snap_recent[512]; /* recent heard words joined   */
};

cp_coach *cp_coach_new(const cp_coach_params *pp, char *err, size_t errn) {
    (void)err; (void)errn;
    cp_coach *c = calloc(1, sizeof *c);
    if (!c) return NULL;
    c->p = *pp;
    if (c->p.target_wpm <= 0) c->p.target_wpm = 150;
    if (c->p.drift_threshold <= 0) c->p.drift_threshold = 5;
    if (c->p.silence_warn_sec <= 0) c->p.silence_warn_sec = 8;
    if (c->p.llm_every_sec <= 0) c->p.llm_every_sec = 45;
    pthread_mutex_init(&c->mu, NULL);
    return c;
}

static void post(cp_coach *c, cp_advice_kind k, const char *text, double at) {
    pthread_mutex_lock(&c->mu);
    c->mbox_seq++;
    c->mailbox.kind = k;
    c->mailbox.at = at;
    c->mailbox.seq = c->mbox_seq;
    snprintf(c->mailbox.text, sizeof c->mailbox.text, "%s", text ? text : "");
    pthread_mutex_unlock(&c->mu);
}

int cp_coach_poll(cp_coach *c, cp_advice *a, int *last_seq) {
    int fresh = 0;
    pthread_mutex_lock(&c->mu);
    if (c->mbox_seq > *last_seq) { *a = c->mailbox; *last_seq = c->mbox_seq; fresh = 1; }
    pthread_mutex_unlock(&c->mu);
    return fresh;
}

cp_advice cp_coach_last(cp_coach *c) {
    cp_advice a;
    pthread_mutex_lock(&c->mu);
    a = c->mailbox;
    pthread_mutex_unlock(&c->mu);
    return a;
}

/* ---------- local rules ---------- */
static void rules(cp_coach *c, const cp_script *s, const cp_pacer *p,
                  float wpm_now, int speaking, double now) {
    /* pace too fast/slow: only meaningful when actually speaking */
    if (speaking && wpm_now > 0) {
        if (wpm_now > c->p.target_wpm * 1.25 &&
            now - c->last_rule_slow_at > 20) {
            c->last_rule_slow_at = now;
            char m[160];
            snprintf(m, sizeof m, "Pace %.0f wpm — slow down (target %.0f)",
                     wpm_now, c->p.target_wpm);
            post(c, CP_ADVICE_SLOW_DOWN, m, now);
        } else if (wpm_now < c->p.target_wpm * 0.6 &&
                   now - c->last_rule_fast_at > 30) {
            c->last_rule_fast_at = now;
            char m[160];
            snprintf(m, sizeof m, "Pace %.0f wpm — you can pick up speed",
                     wpm_now);
            post(c, CP_ADVICE_SPEED_UP, m, now);
        }
    }
    /* drift in tracking mode */
    if (p && s && s->default_mode == CP_MODE_TRACK &&
        p->drifted_by >= c->p.drift_threshold &&
        now - c->last_rule_drift_at > 15) {
        c->last_rule_drift_at = now;
        post(c, CP_ADVICE_RESYNC, "You've drifted from the script — tap to resync", now);
    }
    /* silence mid-show (not paused, not finished) */
    if (p && !p->paused && !cp_pacer_finished(p)) {
        if (!speaking) {
            if (c->t_silent_since == 0) c->t_silent_since = now;
            else if (now - c->t_silent_since > c->p.silence_warn_sec &&
                     now - c->last_rule_silence_at > 45) {
                c->last_rule_silence_at = now;
                post(c, CP_ADVICE_TEXT, "Still rolling — speak or press Space to pause", now);
            }
        } else c->t_silent_since = 0;
    }
}

/* ---------- LLM layer ---------- */
static void build_prompt(const cp_coach *c, const cp_script *s, const cp_pacer *p,
                         float wpm_now, char *user, size_t un) {
    int page = p ? cp_page_of(s, cp_pacer_word_index(p)) : 0;
    double remain = p ? cp_pacer_remaining_seconds(p) : 0;
    snprintf(user, un,
        "state: pos=%.1f/%zu page=%d/%zu wpm=%.0f target=%.0f paused=%d remaining=%.0fs\n"
        "recent transcript: %s\n"
        "Reply with ONE line: either ACTION:<slow_down|speed_up|resync|push_next|hold|none>"
        " or COACH:<max 12 words of encouragement/correction>. Nothing else.",
        p ? p->pos : 0.0, s ? s->nw : (size_t)0, page, s ? s->np : (size_t)0,
        wpm_now, c->p.target_wpm, p ? p->paused : 0, remain,
        c->snap_recent[0] ? c->snap_recent : "(no speech yet)");
}

static cp_advice_kind parse_action(const char *txt) {
    if (strstr(txt, "ACTION:slow_down"))  return CP_ADVICE_SLOW_DOWN;
    if (strstr(txt, "ACTION:speed_up"))   return CP_ADVICE_SPEED_UP;
    if (strstr(txt, "ACTION:resync"))     return CP_ADVICE_RESYNC;
    if (strstr(txt, "ACTION:push_next"))  return CP_ADVICE_PUSH_NEXT;
    if (strstr(txt, "ACTION:hold"))       return CP_ADVICE_HOLD;
    return CP_ADVICE_TEXT;
}

static double mono_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec / 1e9;
}

/* synchronous one-shot (also used by worker) */
int cp_coach_llm_once(cp_coach *c, const cp_script *s, const cp_pacer *p,
                      float wpm_now, cp_advice *out, char *err, size_t errn) {
    char user[1024], reply[512];
    build_prompt(c, s, p, wpm_now, user, sizeof user);
    const char *system =
        "You are a terse teleprompter coach for a live presenter. Prefer none "
        "unless something genuinely needs correction. Never invent script content.";
    if (cp_openai_chat(&c->p.cfg, system, user, 0.2, reply, sizeof reply, err, errn) != 0)
        return -1;
    cp_advice_kind k = parse_action(reply);
    const char *msg = strstr(reply, "COACH:");
    char text[192];
    if (msg) snprintf(text, sizeof text, "%.180s", msg + 6);
    else snprintf(text, sizeof text, "%.180s", reply);
    if (out) { memset(out, 0, sizeof *out); out->kind = k; snprintf(out->text, sizeof out->text, "%s", text); }
    return 0;
}

typedef struct { cp_coach *c; } work_arg;

static void *llm_worker(void *va) {
    work_arg *w = va;
    cp_coach *c = w->c;
    free(w);
    /* take a consistent snapshot under lock, call outside lock. We stored
     * only scalars so the worker never dereferences UI-owned structs. */
    double pos, maxdel, speed, wpm; size_t nw;
    pthread_mutex_lock(&c->mu);
    pos = c->snap_pos; maxdel = c->snap_maxdel; speed = c->snap_speed;
    wpm = c->snap_wpm; nw = c->snap_nw;
    pthread_mutex_unlock(&c->mu);

    cp_pacer pcopy; memset(&pcopy, 0, sizeof pcopy);
    pcopy.pos = pos; pcopy.max_delivered = maxdel; pcopy.speed_wps = speed;
    cp_script scopy; memset(&scopy, 0, sizeof scopy);
    scopy.nw = nw; scopy.np = 1; scopy.default_mode = CP_MODE_VOICE;

    cp_advice a; char err[256];
    if (cp_coach_llm_once(c, &scopy, &pcopy, (float)wpm, &a, err, sizeof err) == 0) {
        post(c, a.kind, a.text, mono_now());
    }
    pthread_mutex_lock(&c->mu);
    c->llm_busy = 0;
    pthread_mutex_unlock(&c->mu);
    return NULL;
}

void cp_coach_update(cp_coach *c, const cp_script *s, const cp_pacer *p,
                     float wpm_now, int speaking, double now) {
    rules(c, s, p, wpm_now, speaking, now);
    /* refresh snapshot + recent transcript window */
    pthread_mutex_lock(&c->mu);
    if (p) { c->snap_pos = p->pos; c->snap_maxdel = p->max_delivered;
             c->snap_speed = p->speed_wps; }
    if (s) { c->snap_nw = s->nw; }
    c->snap_wpm = wpm_now;
    if (c->p.heard_words) {
        /* join last ~20 non-NULL words */
        size_t o = 0; c->snap_recent[0] = 0;
        int n = 0; while (c->p.heard_words[n]) n++;
        int start = n > 20 ? n - 20 : 0;
        for (int i = start; i < n && o < sizeof c->snap_recent - 32; i++)
            o += (size_t)snprintf(c->snap_recent + o, sizeof c->snap_recent - o,
                                  "%s%s", i > start ? " " : "", c->p.heard_words[i]);
    }
    int due = c->p.enable_llm && !c->llm_busy &&
              (now - c->last_llm_at) >= c->p.llm_every_sec &&
              p && !p->paused && !cp_pacer_finished(p);
    if (due) { c->llm_busy = 1; c->last_llm_at = now; }
    pthread_mutex_unlock(&c->mu);
    if (due) {
        work_arg *w = malloc(sizeof *w);
        w->c = c;
        if (pthread_create(&c->th, NULL, llm_worker, w) != 0) {
            pthread_mutex_lock(&c->mu); c->llm_busy = 0; pthread_mutex_unlock(&c->mu);
            free(w);
        }
    }
}

void cp_coach_free(cp_coach *c) {
    if (!c) return;
    pthread_mutex_destroy(&c->mu);
    free(c);
}
