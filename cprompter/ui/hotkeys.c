/* hotkeys.c — see hotkeys.h. ch is the logical key: printable ascii, or
 * CPK_KEY_* pseudo-codes for arrows/space/wheel from UI backends. */
#include "hotkeys.h"

#include <stddef.h>

cp_key_action cpk_lookup(int ch, int ctrl, int alt) {
    if (ctrl || alt) {
        if (ctrl && ch == 'c') return CPK_QUIT;
        return CPK_NONE;
    }
    switch (ch) {
    case 'q': case 'Q':          return CPK_QUIT;
    case CPK_KEY_ESC:            return CPK_QUIT;
    case CPK_KEY_SPACE:          return CPK_PAUSE;
    case '+': case '=':          return CPK_SPEED_UP;
    case '-': case '_':         return CPK_SPEED_DOWN;
    case CPK_WHEEL_UP: case CPK_KEY_UP:    return CPK_SCROLL_FWD;
    case CPK_WHEEL_DOWN: case CPK_KEY_DOWN: return CPK_SCROLL_BACK;
    case ']':                    return CPK_PAGE_NEXT;
    case '[':                    return CPK_PAGE_PREV;
    case CPK_KEY_RIGHT:          return CPK_SCROLL_FWD;
    case CPK_KEY_LEFT:           return CPK_SCROLL_BACK;
    case 'm': case 'M':          return CPK_MODE;
    case 's': case 'S':          return CPK_SAVE;
    case 'r': case 'R':          return CPK_RECAL;
    case ',':                    return CPK_SENS_DOWN;
    case '.':                    return CPK_SENS_UP;
    case CPK_KEY_HOME: case 'g': return CPK_JUMP_HOME;
    case 'f': case 'F':          return CPK_MIRROR;
    case 'o': case 'O':          return CPK_OVERLAY;
    case '?': case 'h':          return CPK_HELP;
    default:                     return CPK_NONE;
    }
}

const char *cpk_name(cp_key_action a) {
    switch (a) {
    case CPK_QUIT: return "quit";
    case CPK_PAUSE: return "pause/resume";
    case CPK_SPEED_UP: return "speed up";
    case CPK_SPEED_DOWN: return "speed down";
    case CPK_SCROLL_FWD: return "scroll forward";
    case CPK_SCROLL_BACK: return "scroll back";
    case CPK_PAGE_NEXT: return "next page";
    case CPK_PAGE_PREV: return "prev page";
    case CPK_MODE: return "cycle guidance mode";
    case CPK_SAVE: return "save .textream";
    case CPK_RECAL: return "recalibrate mic";
    case CPK_SENS_UP: return "mic sensitivity +";
    case CPK_SENS_DOWN: return "mic sensitivity -";
    case CPK_JUMP_HOME: return "jump to start";
    case CPK_MIRROR: return "cycle mirror flip";
    case CPK_OVERLAY: return "cycle overlay style";
    case CPK_HELP: return "help";
    default: return "";
    }
}

const char *cpk_help(void) {
    return
      "keys: Space pause | +/- speed | arrows|wheel scroll | [ ] page |\n"
      "      m mode | f mirror | o overlay | s save | r recalibrate |\n"
      "      , . sensitivity | ? help | q/Esc quit\n";
}
