/* overlay_sdl.h — SDL2 renderer entry points. Two styles (plan §4.3):
 *   fullscreen : opaque black, dedicated display / Sidecar rig
 *   notch      : borderless strip pinned below the top edge, always-on-top
 * The engine owns state; this layer only reads via cp_engine_* getters. */
#ifndef CP_OVERLAY_SDL_H
#define CP_OVERLAY_SDL_H

#include "../core/engine.h"

typedef enum { CP_OVERLAY_FULLSCREEN = 0, CP_OVERLAY_NOTCH = 1 } cp_overlay_style;

/* takes over the main loop until quit; returns exit code.
 * style/mirror are live-cycled with 'o' and 'f' keys inside. */
int cp_overlay_sdl_run(cp_engine *e, int loop, cp_overlay_style style);

#endif
