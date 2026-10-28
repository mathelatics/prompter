/* cPrompter core data model — see docs/audit/cprompter-01-plan.md §2.
 * Script = Pages = Words. One flat word array with page boundaries for O(1)
 * cursor math; pages store [w0, w1) ranges into it. */
#ifndef CP_SCRIPT_H
#define CP_SCRIPT_H

#include <stddef.h>

#define CP_MAX_PAGE_NAME 64

typedef enum {
    CP_MODE_CLASSIC = 0,   /* constant-speed scroll                 */
    CP_MODE_VOICE   = 1,   /* scroll only while speaking            */
    CP_MODE_TRACK   = 2    /* ASR word-tracking (M2: whisper+align) */
} cp_mode;

typedef struct cp_word {
    char   *text;      /* original token, heap-owned                  */
    size_t  n;         /* byte length of text                         */
    int     para;      /* paragraph index within page (for line wrap) */
    float   wpm_prior; /* per-word est. duration weight (1.0 default) */
} cp_word;

typedef struct cp_page {
    char name[CP_MAX_PAGE_NAME];
    int  w0, w1;           /* word range [w0,w1) in script->words   */
    cp_mode mode;          /* per-page override, -1 = use default   */
    double scroll_speed;   /* words/sec, 0.5..8 (classic/voice min) */
    int    countdown_sec;  /* auto-advance pause between pages      */
} cp_page;

typedef struct cp_script {
    cp_word *words;  size_t nw, capw;
    cp_page  *pages; size_t np, capp;
    char     *raw;   /* full original text (kept for save fidelity) */
    /* settings mirroring .textream JSON from the blog post */
    cp_mode  default_mode;
    double   default_speed;   /* words/sec */
    char     lang[16];        /* e.g. "en-US" */
    int      mirror;          /* 0 none, 1 horiz, 2 vert, 3 180deg */
    int      font_family;     /* 0 sans 1 serif 2 mono 3 dyslexic  */
    int      font_size;       /* pt                                   */
    int      highlight_color; /* 0 white 1 yellow 2 green 3 blue 4 pink 5 orange */
} cp_script;

cp_script *cp_script_new(void);
void       cp_script_free(cp_script *s);

/* Tokenize `text` into words+pages. Paragraph breaks (\n\n or lines starting
 * with "# ") start a new page if split_pages, else just a new para. */
int  cp_script_load_text(cp_script *s, const char *text, int split_pages);

/* Word lookup / stats */
const cp_word *cp_word_at(const cp_script *s, size_t i);   /* NULL if oob */
int  cp_page_of(const cp_script *s, size_t word_idx);      /* page index  */
size_t cp_script_count_words(const cp_script *s);          /* == s->nw    */
double cp_script_est_seconds(const cp_script *s, double wps);

#endif /* CP_SCRIPT_H */
