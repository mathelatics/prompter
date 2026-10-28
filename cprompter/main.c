/* cprompter — pure-C teleprompter (see docs/audit/cprompter-01-plan.md)
 * M0 CLI: load script (.txt/.md/.textream), run classic-mode pacer, print
 * word-by-word highlight to terminal; --headless for CI/tests.
 * SDL2 overlay lives in ui/overlay_sdl.c, compiled when CP_HAVE_SDL=1. */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE   /* usleep on glibc with -std=c11 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if CP_HAVE_SDL
#include "ui/overlay_sdl.h"
#endif
#include "core/script.h"
#include "core/pacer.h"
#include "core/engine.h"
#include "format/script_io.h"
#include "format/pptx.h"
#include "ai/coach.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec / 1e9;
}

static char *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    if (n < 0 || n > 64 << 20) { fclose(f); return NULL; }
    char *buf = malloc(n + 1);
    if (!buf || fread(buf, 1, n, f) != (size_t)n) { free(buf); fclose(f); return NULL; }
    buf[n] = 0; fclose(f);
    if (len_out) *len_out = (size_t)n;
    return buf;
}

typedef struct {
    const char *path;
    double speed;        /* words/sec */
    int     headless;
    int     loop;
    int     show_help;
    cp_mode mode;        /* override (-1 none) */
    int     coach;       /* M5: 0 rules-off, 1 local rules, 2 + LLM advisor */
    float   target_wpm;  /* coach ideal pace                              */
} opts;

static void usage(void) {
    fprintf(stderr,
      "cprompter — pure-C teleprompter\n"
      "usage: cprompter [script.txt|script.md|script.textream] [options]\n"
      "  --speed <wps>       scroll speed, words/sec (default 3.0)\n"
      "  --mode <classic|voice|track>   guidance mode override\n"
      "  --headless          no curses/SDL, plain stdout (for CI/ssh)\n"
      "  --loop              restart at end\n"
      "  --save-out <path>   write loaded script as .textream and exit\n"
      "  --coach <rules|llm>  M5 AI coach (rules = offline heuristics;\n"
      "                      llm adds OpenAI-compatible advisor via\n"
      "                      CCP_BASE_URL / CCP_MODEL / CCP_API_KEY)\n"
      "  --target-wpm <n>    coach's ideal speaking rate (default 150)\n"
      "  -h, --help          this message\n");
}

int main(int argc, char **argv) {
    opts o = { "samples/demo.txt", 3.0, 0, 0, 0, (cp_mode)-1, 0, 150 };
    const char *save_out = NULL;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--speed") && i + 1 < argc) o.speed = atof(argv[++i]);
        else if (!strcmp(a, "--headless")) o.headless = 1;
        else if (!strcmp(a, "--loop")) o.loop = 1;
        else if (!strcmp(a, "--coach") && i + 1 < argc) {
            const char *c = argv[++i];
            o.coach = !strcmp(c, "llm") ? 2 : !strcmp(c, "rules") ? 1 : 0;
        }
        else if (!strcmp(a, "--target-wpm") && i + 1 < argc) o.target_wpm = (float)atof(argv[++i]);
        else if (!strcmp(a, "--save-out") && i + 1 < argc) save_out = argv[++i];
        else if (!strcmp(a, "--mode") && i + 1 < argc) {
            const char *m = argv[++i];
            o.mode = !strcmp(m, "voice") ? CP_MODE_VOICE
                   : !strcmp(m, "track") ? CP_MODE_TRACK : CP_MODE_CLASSIC;
        }
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); return 0; }
        else if (a[0] != '-') o.path = a;
        else { fprintf(stderr, "unknown option: %s\n", a); usage(); return 2; }
    }
    if (o.speed < 0.5 || o.speed > 8.0) {
        fprintf(stderr, "speed must be 0.5..8 words/sec (Textream range)\n");
        return 2;
    }

    /* ---- load ---- */
    cp_script *s = cp_script_new();

    size_t rl = strlen(o.path);
    if (rl >= 5 && !strcmp(o.path + rl - 5, ".pptx")) {
        /* M4: PowerPoint presenter-notes import (needs system unzip) */
        char err[128];
        if (cp_script_load_pptx(s, o.path, err, sizeof err) != 0) {
            fprintf(stderr, "pptx import: %s\n", err);
            cp_script_free(s);
            return 1;
        }
    } else {
    size_t len = 0;
    char *raw = read_file(o.path, &len);
    if (!raw) { fprintf(stderr, "cannot read script: %s\n", o.path); return 1; }

    if (rl >= 9 && !strcmp(o.path + rl - 9, ".textream")) {
        char err[128];
        if (cp_script_load_textream(s, o.path, err, sizeof err) != 0) {
            fprintf(stderr, "textream parse: %s\n", err);
            /* fall back: treat file as raw text */
            cp_script_load_text(s, raw, 1);
        }
    } else {
        cp_script_load_text(s, raw, 1);
    }
    free(raw);
    }
    if (o.mode != (cp_mode)-1) s->default_mode = o.mode;
    s->default_speed = o.speed;

    if (save_out) {
        int rc = cp_script_save_textream(s, save_out);
        printf("saved %s (%zu pages, %zu words)\n", save_out, s->np, s->nw);
        cp_script_free(s);
        return rc == 0 ? 0 : 1;
    }

    printf("cprompter: %zu page(s), %zu words, ~%.0f min @ %.1f w/s [%s]\n",
           s->np, s->nw, cp_script_est_seconds(s, o.speed) / 60.0, o.speed,
           s->default_mode == CP_MODE_TRACK ? "track" :
           s->default_mode == CP_MODE_VOICE ? "voice" : "classic");

    cp_pacer p;
    cp_pacer_init(&p, s, o.speed);

#if CP_HAVE_SDL
    if (!o.headless && isatty(1)) {
        /* engine owns audio/VAD/WPM/track; overlay owns the UI clock */
        cp_engine_params ep = {0};
        ep.dev = NULL;
        ep.enable_audio = (s->default_mode == CP_MODE_VOICE);
        ep.track.backend = (s->default_mode == CP_MODE_TRACK)
                           ? CP_TRACK_WHISPER : CP_TRACK_NONE;
        ep.track.lang = s->lang[0] ? s->lang : "en";
        char eerr[256] = "";
        cp_engine *eng = cp_engine_new(s, &p, &ep, eerr, sizeof eerr);
        if (!eng) {
            fprintf(stderr, "engine: %s (falling back to classic clock)\n", eerr);
            ep.enable_audio = 0;
            ep.track.backend = CP_TRACK_NONE;
            eng = cp_engine_new(s, &p, &ep, eerr, sizeof eerr);
            if (!eng) { fprintf(stderr, "engine init failed: %s\n", eerr);
                        cp_script_free(s); return 1; }
        }
        int rc = cp_overlay_sdl_run(eng, o.loop, CP_OVERLAY_FULLSCREEN);
        cp_engine_free(eng);
        cp_script_free(s);
        return rc;
    }
#endif
    /* ---- headless / terminal loop: one line per frame, ANSI clear ---- */
    /* M5 coach (rules always local; --coach llm adds OpenAI-compatible advisor) */
    cp_coach *coach = NULL;
    int coach_seq = 0;
    if (o.coach) {
        cp_coach_params cpar; memset(&cpar, 0, sizeof cpar);
        cpar.enable_llm = (o.coach >= 2);
        cpar.target_wpm = o.target_wpm;
        if (cpar.enable_llm && cp_openai_cfg_from_env(&cpar.cfg) != 0) {
            fprintf(stderr, "coach: bad CCP_BASE_URL, falling back to rules\n");
            cpar.enable_llm = 0;
        }
        char cerr[256] = "";
        coach = cp_coach_new(&cpar, cerr, sizeof cerr);
        if (!coach) fprintf(stderr, "coach init failed: %s\n", cerr);
        else printf("coach: %s (target %.0f wpm)\n",
                    cpar.enable_llm ? "rules+llm" : "rules", cpar.target_wpm);
    }
    double t0 = now_sec(), last_print = 0;
    int prev_word = -1;
    int pg = 0;
    while (!cp_pacer_finished(&p) || o.loop) {
        double now = now_sec() - t0;
        double dt = now - (last_print ? last_print : now);
        if (dt > 0.05) dt = 0.05;                 /* cap on slow terminals */
        cp_pacer_tick(&p, now, dt);
        last_print = now;

        if (coach) {
            cp_advice a;
            cp_coach_update(coach, s, &p, (float)p.measured_wpm,
                            p.vad_active, now);
            if (cp_coach_poll(coach, &a, &coach_seq)) {
                /* apply auto-director actions the same way a human would */
                switch (a.kind) {
                    case CP_ADVICE_SLOW_DOWN:
                        cp_pacer_set_speed(&p, p.speed_wps * 0.85); break;
                    case CP_ADVICE_SPEED_UP:
                        cp_pacer_set_speed(&p, p.speed_wps * 1.15); break;
                    case CP_ADVICE_PUSH_NEXT:
                        if ((size_t)pg + 1 < s->np)
                            cp_pacer_jump(&p, s->pages[pg + 1].w0, 1);
                        break;
                    case CP_ADVICE_HOLD:
                        cp_pacer_set_paused(&p, 1); break;
                    default: break;
                }
                fprintf(stderr, "\r\033[K\033[36m[coach] %s\033[0m\n", a.text);
            }
        }

        size_t wi = cp_pacer_word_index(&p);
        if ((int)wi != prev_word) {
            prev_word = (int)wi;
            /* render a small window of context like Textream's panel */
            pg = cp_page_of(s, wi);
            printf("\r\033[K");                     /* clear line            */
            int lo = (int)wi - 3, hi = (int)wi + 7;
            if (lo < 0) lo = 0;
            for (int i = lo; i < hi && i < (int)s->nw; i++) {
                if (i == (int)wi) printf("\033[1;33m%s\033[0m ", s->words[i].text);
                else if (i < (int)wi) printf("\033[2m%s\033[0m ", s->words[i].text);
                else printf("%s ", s->words[i].text);
            }
            fflush(stdout);
        }
        if (cp_pacer_finished(&p)) {
            if (!o.loop) break;
            cp_pacer_init(&p, s, o.speed);
            t0 = now_sec(); last_print = 0; prev_word = -1;
            printf("\n--- restart (page: %s) ---\n", pg >= 0 ? s->pages[pg].name : "");
        }
        usleep(8 * 1000);                          /* ~125 fps tick ceiling */
    }
    printf("\n[done] delivered %d/%zu words, max_delivered=%.1f\n",
           (int)cp_pacer_word_index(&p), s->nw, p.max_delivered);
    cp_script_free(s);
    return 0;
}
