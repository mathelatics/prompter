/* ai/json — tiny JSON value extractor, enough for OpenAI-compatible API
 * responses (chat completions, /v1/models). Not a validator: it finds the
 * first match and unescapes strings. Deliberately ~200 LoC; correctness is
 * exercised by unit tests against real-shaped payloads. */
#ifndef CP_JSON_H
#define CP_JSON_H

#include <stddef.h>

/* Find the value of top-level-or-anywhere key "key" and copy the string
 * value into out (JSON-unescaped). Returns 1 on success, 0 if not found or
 * not a string. Searches every occurrence; use cp_json_find_path for nested
 * dotted paths like "choices.0.message.content". */
int cp_json_get_string(const char *json, const char *key,
                       char *out, size_t outn);

/* Dotted path through objects/arrays ("a.b.0.c"). Numeric segments index
 * arrays. Returns 1 on success. */
int cp_json_find_path(const char *json, const char *path,
                      char *out, size_t outn);

/* Extract a number (int) at dotted path; returns def if absent. */
double cp_json_get_number(const char *json, const char *path, double def);

#endif
