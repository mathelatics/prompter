/* .textream save format — JSON shape from cprompter-01-plan.md / blog:
 * { "version":"1.0",
 *   "pages":[ {"id":1,"content":"...", "settings":{"guidanceMode":
 *              "wordTracking"|"classic"|"voiceActivated","scrollSpeed":3.0,
 *              "speechLanguage":"en-US"}} ],
 *   "metadata":{...}, "settings":{ "mirror":0,"fontFamily":"sans",
 *              "fontSize":24,"highlightColor":"yellow" } }
 *
 * Hand-rolled minimal reader/writer (jsmn-style scanner, ~this file only).
 * Swappable for parson later; keeps M0 build dependency-free. */
#ifndef CP_SCRIPT_IO_H
#define CP_SCRIPT_IO_H

#include "../core/script.h"

/* returns 0 on success; err buffer gets a message on failure */
int cp_script_save_textream(const cp_script *s, const char *path);
int cp_script_load_textream(cp_script *s, const char *path, char *err, size_t errn);

/* plain-text convenience (whole file = one script, "# " headings -> pages) */
int cp_script_save_txt(const cp_script *s, const char *path);

#endif
