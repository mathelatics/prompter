/* pptx.h — M4 PowerPoint presenter-notes import. */
#ifndef CP_PPTX_H
#define CP_PPTX_H

#include <stddef.h>
#include "script_io.h"   /* brings in cp_script */

/* Load presenter notes from a .pptx file into s (one page per slide).
 * Requires system `unzip`. Returns 0 on success, -1 + err message otherwise. */
int cp_script_load_pptx(cp_script *s, const char *path, char *err, size_t errn);

/* Exposed for unit testing: concatenate all <a:t> runs from an XML buffer,
 * inserting '\n' at </a:p> paragraph boundaries, entities decoded. malloc'd. */
char *cp_pptx_extract_at_texts(const char *xml, size_t len);

#endif /* CP_PPTX_H */
