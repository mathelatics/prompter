/* hotkeys — single source of truth for key bindings (terminal + SDL share).
 * Textream-inspired: Space=pause, wheel/±=speed, arrows=scroll, [ ]=page,
 * m=cycle mode, s=save .textream, r=recalibrate VAD, +/- sens, q=quit. */
#ifndef CP_HOTKEYS_H
#define CP_HOTKEYS_H

typedef enum {
    CPK_NONE = 0, CPK_QUIT, CPK_PAUSE, CPK_SPEED_UP, CPK_SPEED_DOWN,
    CPK_SCROLL_FWD, CPK_SCROLL_BACK, CPK_PAGE_NEXT, CPK_PAGE_PREV,
    CPK_MODE, CPK_SAVE, CPK_RECAL, CPK_SENS_UP, CPK_SENS_DOWN,
    CPK_JUMP_HOME, CPK_HELP, CPK_MIRROR, CPK_OVERLAY
} cp_key_action;

/* pseudo key codes for non-printable keys (UI backends map to these) */
#define CPK_KEY_SPACE   0x100
#define CPK_KEY_LEFT    0x101
#define CPK_KEY_RIGHT   0x102
#define CPK_KEY_UP      0x103
#define CPK_KEY_DOWN    0x104
#define CPK_KEY_HOME    0x105
#define CPK_WHEEL_UP    0x106
#define CPK_WHEEL_DOWN  0x107
#define CPK_KEY_ESC     0x108

/* map a character (or SDL keycode translated to ascii-ish) to an action.
 * up=1 means the event is on key release (ignored here). */
cp_key_action cpk_lookup(int ch, int ctrl, int alt);

const char *cpk_help(void);
const char *cpk_name(cp_key_action a);

#endif
