/* test_core.c — M0 unit tests: tokenizer, pacer modes, .textream round-trip.
 * Build via `make test`. No framework; CHECK macro with counters. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../core/script.h"
#include "../core/pacer.h"
#include "../format/script_io.h"
#include "../audio/align.h"
#include "../audio/track.h"

static int fails = 0, checks = 0;
#define CHECK(cond) do { checks++; if (!(cond)) { \
    printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); fails++; } } while (0)

static void test_tokenizer(void) {
    cp_script *s = cp_script_new();
    const char *txt =
        "# Intro\nHello world, this is page one.\n\nSecond paragraph here.\n"
        "# Second page\nOnly one line.\n";
    CHECK(cp_script_load_text(s, txt, 1) == 0);
    CHECK(s->np == 2);
    CHECK(!strcmp(s->pages[0].name, "Intro"));
    CHECK(!strcmp(s->pages[1].name, "Second page"));
    /* words: Hello world, this is page one. (6) + Second paragraph here. (3) = 9 */
    CHECK(s->pages[0].w1 - s->pages[0].w0 == 9);
    CHECK(s->pages[1].w1 - s->pages[1].w0 == 3);
    CHECK(s->nw == 12);
    /* paragraphs differ across blank line */
    CHECK(s->words[5].para != s->words[6].para);
    /* UTF-8 word survives */
    cp_script_load_text(s, "caf\u00e9 na\u00efve \xe2\x80\x94 dash caf\xc3\xa9", 0);
    CHECK(s->nw == 5);   /* é kept attached; em-dash standalone token */
    CHECK(!strcmp(s->words[0].text, "caf\xc3\xa9"));
    cp_script_free(s);
}

static void test_pacer_classic(void) {
    cp_script *s = cp_script_new();
    cp_script_load_text(s, "one two three four five six seven eight nine ten", 0);
    cp_pacer p;
    cp_pacer_init(&p, s, 2.0);           /* 2 words/sec */
    double t = 0;
    for (int i = 0; i < 10; i++) { t += 0.5; cp_pacer_tick(&p, t, 0.5); }
    CHECK(fabs(p.pos - 10.0) < 1e-9);
    CHECK(cp_pacer_finished(&p));
    CHECK(p.max_delivered >= 9.99);
    /* manual scroll then pause behavior */
    cp_pacer_init(&p, s, 2.0);
    cp_pacer_scroll_by(&p, 3.0);
    CHECK(p.pos == 3.0);
    cp_pacer_set_paused(&p, 1);
    t += 1.0; cp_pacer_tick(&p, t, 1.0);
    CHECK(p.pos == 3.0);                 /* frozen while paused */
    cp_pacer_set_paused(&p, 0);
    t += 1.0; cp_pacer_tick(&p, t, 1.0);
    CHECK(p.pos == 5.0);
    /* backward jump only when allowed */
    cp_pacer_jump(&p, 1.0, 0);
    CHECK(p.pos == 5.0);
    cp_pacer_jump(&p, 1.0, 1);
    CHECK(p.pos == 1.0);
    CHECK(p.max_delivered == 5.0);       /* high-water mark preserved */
    cp_script_free(s);
}

static void test_pacer_voice_track(void) {
    cp_script *s = cp_script_new();
    cp_script_load_text(s, "a b c d e f g h i j k l", 0);
    s->default_mode = CP_MODE_VOICE;
    cp_pacer p;
    cp_pacer_init(&p, s, 2.0);
    double t = 0;
    cp_pacer_feed_vad(&p, 0, 0);
    t += 1.0; cp_pacer_tick(&p, t, 1.0);
    CHECK(p.pos == 0.0);                 /* silence -> no movement */
    cp_pacer_feed_vad(&p, 1, 120.0);     /* speaking, 120 wpm observed */
    t += 1.0; cp_pacer_tick(&p, t, 1.0);
    CHECK(fabs(p.pos - 2.0) < 1e-9);     /* max(2 wps floor, 2 wps obs)=2 */
    cp_pacer_feed_vad(&p, 1, 240.0);     /* faster speech wins */
    p.measured_wpm = 240;                /* force EMA settled */
    t += 1.0; cp_pacer_tick(&p, t, 1.0);
    CHECK(p.pos > 3.5);                  /* 4 wps now */

    /* track mode: forward ease toward align target, never back */
    s->default_mode = CP_MODE_TRACK;
    cp_pacer_init(&p, s, 2.0);
    cp_pacer_feed_align(&p, 6.0);
    double prev = -1; int monotone = 1;
    for (int i = 0; i < 600; i++) {          /* 60s sim: catch-up cap ~5 w/s */
        t += 0.1; cp_pacer_tick(&p, t, 0.1);
        if (p.pos < prev - 1e-12) monotone = 0;
        prev = p.pos;
    }
    CHECK(monotone);
    CHECK(p.pos >= 5.9);                 /* converged near target */
    /* bounded backtrack (plan §4.1): small re-read corrections allowed,
     * large backward ASR jumps ignored (never below max_delivered-6).
     * Fresh pacer + tick so pos advances toward the target; then a jump
     * back to 2.0 must be rejected (floor = max_delivered(6)-6 = 0 < 2),
     * while 5.0 is forward-of-pos and accepted. */
    cp_pacer_init(&p, s, 2.0);
    cp_pacer_feed_align(&p, 6.0);
    CHECK(p.track_target == 6.0);            /* forward from 0 accepted   */
    for (int i = 0; i < 80; i++) { t += 0.1; cp_pacer_tick(&p, t, 0.1); }
    CHECK(p.pos > 2.5 && p.pos < 6.0);       /* easing, capped catch-up   */
    cp_pacer_feed_align(&p, 2.0);            /* behind pos -> rejected    */
    CHECK(p.track_target == 6.0);
    cp_pacer_feed_align(&p, 5.0);            /* ahead of pos -> accepted  */
    CHECK(fabs(p.track_target - 5.0) < 1e-9);
    cp_pacer_feed_align(&p, 0.0);            /* far behind -> rejected    */
    CHECK(fabs(p.track_target - 5.0) < 1e-9);
    cp_pacer_feed_align(&p, 11.5);           /* forward always accepted   */
    CHECK(fabs(p.track_target - 11.5) < 1e-9);
    cp_script_free(s);
}

static void test_textream_roundtrip(void) {
    cp_script *s = cp_script_new();
    const char *txt =
        "# Sponsor\nRead this sponsor block slowly and clearly.\n\nThen thank them warmly.\n"
        "# Outro\nGoodbye and subscribe.\n";
    cp_script_load_text(s, txt, 1);
    s->default_speed = 2.5;
    s->highlight_color = 1;   /* yellow */
    s->font_family = 3;       /* dyslexic */
    s->mirror = 1;
    snprintf(s->lang, sizeof s->lang, "en-GB");
    if (s->np >= 2) { s->pages[1].mode = CP_MODE_VOICE; s->pages[1].scroll_speed = 1.5; }

    const char *path = "/tmp/cp_rt.textream";
    CHECK(cp_script_save_textream(s, path) == 0);

    cp_script *r = cp_script_new();
    char err[128];
    CHECK(cp_script_load_textream(r, path, err, sizeof err) == 0);
    CHECK(r->np == s->np);
    CHECK(r->nw == s->nw);
    CHECK(!strcmp(r->lang, "en-GB"));
    CHECK(r->highlight_color == 1 && r->font_family == 3 && r->mirror == 1);
    CHECK(!strcmp(r->pages[0].name, "Sponsor"));
    CHECK(!strcmp(r->words[0].text, "Read"));
    if (r->np >= 2) {
        CHECK(r->pages[1].mode == CP_MODE_VOICE);
        CHECK(fabs(r->pages[1].scroll_speed - 1.5) < 1e-6);
    }
    /* content fidelity: same word sequence */
    for (size_t i = 0; i < s->nw; i++)
        CHECK(!strcmp(s->words[i].text, r->words[i].text));
    cp_script_free(s); cp_script_free(r);
}

/* ---- M2: word aligner --------------------------------------------------- */
static void test_align_basic(void) {
    cp_script *s = cp_script_new();
    cp_script_load_text(s, "the quick brown fox jumps over the lazy dog", 1);
    cp_align *a = cp_align_new(s);

    /* exact in-order speech advances one word per recognized token */
    const char *say[] = {"The", "quick,", "brown", "fox"};
    for (int i = 0; i < 4; i++) {
        cp_align_result r = cp_align_feed(a, say[i]);
        CHECK(r.matched == 1);
        CHECK(cp_align_pos(a) == (size_t)(i + 1));
    }

    /* fillers are dropped silently */
    cp_align_result rf = cp_align_feed(a, "um");
    CHECK(rf.filler_skipped == 1 && rf.matched == 0);
    CHECK(cp_align_pos(a) == 4);

    /* skipped words: speaker rushed past "jumps over" -> forward search finds
     * the next script word ("the", idx 6); one advance per spoken token. */
    cp_align_result rs = cp_align_feed(a, "the");
    CHECK(rs.matched == 1);
    CHECK(cp_align_pos(a) == 7);          /* jumped ahead to second "the" */

    /* re-read/stutter: repeat a just-delivered word; no further "the" exists
     * ahead, so it snaps back to the boundary right after it (idx 6 -> pos 7) */
    cp_align_result rb = cp_align_feed(a, "the");
    CHECK(rb.matched == 1);
    CHECK(cp_align_pos(a) == 7);

    /* garbage streak raises drift after AL_GIVEUP misses */
    cp_align_reset(a, 0);
    for (int i = 0; i < 31; i++) cp_align_feed(a, "zzzqx");
    CHECK(cp_align_drift(a) >= 30);
    CHECK(cp_align_pos(a) == 0);          /* never advanced on pure noise */

    /* fuzzy match: typo within edit distance 1 still tracks (long words only;
     * short/function words require exact normalized match by design) */
    cp_align_reset(a, 0);
    cp_align_feed(a, "teh");                  /* "the" is <=3 chars: exact-only */
    CHECK(cp_align_pos(a) == 0);              /* rejected by design             */
    cp_align_feed(a, "quickk");               /* ~ "quick" (insertion, dist 1)  */
    CHECK(cp_align_pos(a) == 2);              /* matched via fuzzy fast path    */
    cp_align_free(a); cp_script_free(s);
}

/* helpers exposed by align.c — verify directly too */
static void test_align_helpers(void) {
    char buf[64];
    /* cp_norm_word returns the number of normalized chars written */
    CHECK(cp_norm_word("Hello,", buf, sizeof buf) == 5);
    CHECK(strcmp(buf, "hello") == 0);
    CHECK(cp_is_filler("uh"));
    CHECK(!cp_is_filler("cat"));
    CHECK(cp_fuzzy_eq("color", "colour"));       /* Levenshtein 1 */
    CHECK(cp_fuzzy_eq("teleprompter", "telepromp")); /* ASR truncation */
    CHECK(!cp_fuzzy_eq("cat", "dog"));
}

/* ---- M2: mock ASR backend -> pacer pipeline ------------------------------ */
static void test_track_mock_pipeline(void) {
    static const char *stream[] = {"the", "quick", "brown", "fox", NULL};
    cp_track_params tp; memset(&tp, 0, sizeof tp);
    tp.backend   = CP_TRACK_MOCK;
    tp.mock_words = stream;
    tp.mock_wps   = 50.0;                    /* fast feed for the test */
    char err[128];
    cp_track *t = cp_track_open(&tp, err, sizeof err);
    CHECK(t != NULL);

    cp_script *s = cp_script_new();
    cp_script_load_text(s, "the quick brown fox jumps over the lazy dog", 1);
    s->default_mode = CP_MODE_TRACK;         /* activate track semantics   */
    cp_pacer p; cp_pacer_init(&p, s, 2.0);
    cp_align *a = cp_align_new(s);

    const char *words[16]; int total = 0; double now = 0;
    /* real-time loop: poll at ~20Hz (wall clock), feed aligner, tick pacer */
    for (int iter = 0; iter < 400 && total < 4; iter++) {
        size_t n = cp_track_poll(t, words, 16);
        for (size_t i = 0; i < n; i++) {
            cp_pacer_feed_align(&p, (double)cp_align_feed(a, words[i]).script_idx);
            total++;
        }
        struct timespec ts = { 0, 50 * 1000 * 1000 }; nanosleep(&ts, NULL);
        now += 0.05; cp_pacer_tick(&p, now, 0.05);
    }
    CHECK(total == 4);                       /* all mock words drained */
    CHECK(cp_align_pos(a) == 4);             /* aligned at "jumps" boundary */
    CHECK(p.track_target == 4.0);
    for (int i = 0; i < 200; i++) { now += 0.05; cp_pacer_tick(&p, now, 0.05); }
    CHECK(p.pos >= 3.9);                     /* display eased onto target */
    cp_align_free(a); cp_script_free(s); cp_track_close(t);
}

int main(void) {
    test_tokenizer();
    test_pacer_classic();
    test_pacer_voice_track();
    test_textream_roundtrip();
    test_align_basic();
    test_align_helpers();
    test_track_mock_pipeline();
    printf("%d checks, %d failures\n", checks, fails);
    return fails ? 1 : 0;
}
