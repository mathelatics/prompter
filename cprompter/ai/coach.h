/* ai/coach — M5 AI coach / auto-director.
 * Two layers, per cprompter-01-plan.md §"coach":
 *   1) LOCAL RULES (always on, zero-latency, no network): pace too fast/slow,
 *      tracker drift > threshold, long silence mid-show, page nearing end.
 *   2) LLM ADVISOR (OpenAI-compatible, throttled): sends a compact state
 *      summary + recent transcript window, asks for one short coaching line
 *      or an auto-direct action ("push_next", "slow_down", "hold"). Runs in
 *      a worker thread so the render loop never blocks; results land in a
 *      single-slot mailbox consumed by engine/UI at their pace.
 * The coach is READ-ONLY toward the pacer except via explicit actions the
 * host applies (jump/pause/speed) — same API surface director mode uses. */
#ifndef CP_COACH_H
#define CP_COACH_H

#include <stddef.h>
#include "../core/script.h"
#include "../core/pacer.h"
#include "openai.h"

typedef enum {
    CP_ADVICE_NONE = 0,
    CP_ADVICE_TEXT,        /* display msg (<= 160 bytes utf8)             */
    CP_ADVICE_SLOW_DOWN,   /* host should reduce speed ~15%               */
    CP_ADVICE_SPEED_UP,    /* host should raise speed ~15%                */
    CP_ADVICE_RESYNC,      /* tap-to-resync hint (drift too big)          */
    CP_ADVICE_PUSH_NEXT,   /* auto-director: jump to next page            */
    CP_ADVICE_HOLD         /* pause scroll until manual resume            */
} cp_advice_kind;

typedef struct {
    cp_advice_kind kind;
    char text[192];
    double at;             /* monotonic timestamp of generation           */
    int seq;               /* increments per new advice; UI dedupes       */
} cp_advice;

typedef struct cp_coach cp_coach;

typedef struct {
    int   enable_llm;          /* 0 -> rules only                          */
    cp_openai_cfg cfg;         /* base_url/model/token                     */
    double llm_every_sec;      /* min gap between LLM calls (default 45)   */
    float  target_wpm;         /* ideal speaking rate (default 150)        */
    int    drift_threshold;    /* consecutive misses before resync (def 5) */
    double silence_warn_sec;   /* silent-but-unpaused warning (default 8)  */
    const char **heard_words;  /* optional: NULL-term array of ASR words
                                * coach reads each tick (mock/test hook)   */
} cp_coach_params;

cp_coach *cp_coach_new(const cp_coach_params *p, char *err, size_t errn);
void      cp_coach_free(cp_coach *c);

/* Call once per UI frame with current state. Non-blocking: evaluates local
 * rules immediately; schedules LLM work on a background thread when due. */
void cp_coach_update(cp_coach *c, const cp_script *s, const cp_pacer *p,
                     float wpm_now, int speaking, double now_sec);

/* Poll newest advice not yet seen (seq > last_seq). Returns 1 and fills a
 * if fresh advice exists. Thread-safe. */
int  cp_coach_poll(cp_coach *c, cp_advice *a, int *last_seq);

/* Latest rule-evaluation snapshot for tests/debug (no side effects). */
cp_advice cp_coach_last(cp_coach *c);

/* For tests: run one synchronous LLM round-trip right now. */
int cp_coach_llm_once(cp_coach *c, const cp_script *s, const cp_pacer *p,
                      float wpm_now, cp_advice *out, char *err, size_t errn);

#endif
