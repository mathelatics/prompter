/* align — word tracker for M2 Word-Tracking mode (plan §4.2, cprompter-01).
 * Turns a stream of recognized words (from whisper.cpp in track.c, or from
 * any external source: cloud ASR via the agent HTTP layer, a test harness)
 * into a monotone cursor position inside cp_script's flat word array.
 *
 * Algorithm (subsequence alignment + local edit fallback):
 *   - fast path: next expected script word matched (exact / normalized /
 *     phonetic-ish fuzzy) -> advance cursor 1;
 *   - lookahead: presenter skipped ahead (<= AL_MAX_LOOK words) -> jump fwd;
 *   - re-read/backtrack (<= AL_MAX_BACK words) -> allow small backward move
 *     (Textream tolerates stutters/repeats) but NEVER below max_delivered+1
 *     unless explicitly allowed, keeping read-lock semantics intact;
 *   - filler/hesitation words ("um", "uh", "so", ...) are dropped silently;
 *   - if nothing matches within the window, stay put and count misses; after
 *     AL_GIVEUP misses we flag drift so the UI/coach can hint "tap to resync".
 *
 * Pure C, no deps, fully unit-testable with cp_align_feed() synthetic streams.
 */
#ifndef CP_ALIGN_H
#define CP_ALIGN_H

#include <stddef.h>
#include "../core/script.h"

typedef struct cp_align cp_align;

typedef struct {
    int   advanced;      /* words the cursor moved this call (can be <0)   */
    int   matched;       /* 1 = some match happened                        */
    int   filler_skipped;/* 1 = input word was a filler, ignored           */
    size_t script_idx;   /* current cursor (floor) after update            */
    int   drift;         /* consecutive unmatched recognition words        */
} cp_align_result;

cp_align *cp_align_new(const cp_script *s);
void      cp_align_free(cp_align *a);

/* reset cursor to a script word index (e.g. on page change or manual jump) */
void      cp_align_reset(cp_align *a, size_t start_word_idx);

/* feed one recognized word (UTF-8, may contain punctuation). Returns result. */
cp_align_result cp_align_feed(cp_align *a, const char *word);

/* state queries */
size_t cp_align_pos(const cp_align *a);          /* fractional-free cursor  */
int    cp_align_drift(const cp_align *a);        /* miss streak             */
const char *cp_align_last_heard(const cp_align *a); /* debug/UI ticker     */

/* helpers exposed for tests */
int  cp_norm_word(const char *in, char *out, size_t outn); /* lowercase, strip punct/diacritics-ish */
int  cp_is_filler(const char *norm);                       /* 1 if hesitation/filler token          */
int  cp_fuzzy_eq(const char *a, char *b);                  /* 1 if Levenshtein<=1 or prefix>=4      */

#endif /* CP_ALIGN_H */
