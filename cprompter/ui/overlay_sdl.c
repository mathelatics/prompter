/* overlay_sdl.c — M1 SDL2 renderer: fullscreen + notch styles, live mirror
 * flip (H/V/180 via SDL_RenderCopyEx), status bar with mode/WPM/mic level,
 * page-aware rendering, hotkeys routed through ui/hotkeys.c so terminal and
 * GUI bindings never drift apart. SDL_ttf optional (block-meter fallback). */
#include "overlay_sdl.h"
#include "hotkeys.h"

#include <SDL.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static const Uint32 hl_colors[6] = {
    0xFFFFFFFF, /* white   */
    0xFFFFFF00, /* yellow  */
    0xFF00FF80, /* green   */
    0xFF40A0FF, /* blue    */
    0xFFFF69B4, /* pink    */
    0xFFFF8C00  /* orange  */
};

static void sdl_key_to_logical(const SDL_KeyboardEvent *k, int *ch, int *ctrl, int *alt) {
    SDL_Keycode s = k->keysym.sym;
    *ctrl = (k->keysym.mod & KMOD_CTRL) != 0;
    *alt  = (k->keysym.mod & KMOD_ALT)  != 0;
    switch (s) {
    case SDLK_SPACE: *ch = CPK_KEY_SPACE;  return;
    case SDLK_LEFT:  *ch = CPK_KEY_LEFT;   return;
    case SDLK_RIGHT: *ch = CPK_KEY_RIGHT;  return;
    case SDLK_UP:    *ch = CPK_KEY_UP;     return;
    case SDLK_DOWN:  *ch = CPK_KEY_DOWN;   return;
    case SDLK_HOME:  *ch = CPK_KEY_HOME;   return;
    case SDLK_ESCAPE:*ch = CPK_KEY_ESC;    return;
    default: break;
    }
    if (s >= SDLK_a && s <= SDLK_z) *ch = 'a' + (s - SDLK_a);
    else if (s == SDLK_PLUS || s == SDLK_EQUALS) *ch = '+';
    else if (s == SDLK_MINUS) *ch = '-';
    else if (s == SDLK_LEFTBRACKET) *ch = '[';
    else if (s == SDLK_RIGHTBRACKET) *ch = ']';
    else if (s == SDLK_COMMA) *ch = ',';
    else if (s == SDLK_PERIOD) *ch = '.';
    else if (s == SDLK_SLASH) *ch = '?';
    else if (s == SDLK_q) *ch = 'q';
    else *ch = (int)s;
}

typedef struct {
    cp_engine       *e;
    const cp_script *s;
    const cp_pacer  *p;
    cp_overlay_style style;
    int              mirror;      /* 0 none 1 horiz 2 vert 3 180 */
    int              running;
    char             toast[80];
    double           toast_until;
#ifdef CP_HAVE_SDL_TTF
    TTF_Font        *font, *font_small;
#endif
} view;

static void toast(view *v, double now, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vsnprintf(v->toast, sizeof v->toast, fmt, ap);
    va_end(ap);
    v->toast_until = now + 1.5;
}

static void apply_style(SDL_Window *win, cp_overlay_style st) {
    if (st == CP_OVERLAY_NOTCH) {
        int dpy = SDL_GetWindowDisplayIndex(win);
        SDL_Rect b; SDL_GetDisplayBounds(dpy, &b);
        SDL_SetWindowBordered(win, SDL_FALSE);
        SDL_SetWindowAlwaysOnTop(win, SDL_TRUE);
        SDL_SetWindowSize(win, b.w * 3 / 5, 140);
        SDL_SetWindowPosition(win, b.x + (b.w - b.w * 3 / 5) / 2, b.y + 8);
        SDL_SetWindowOpacity(win, 0.92f);
    } else {
        SDL_SetWindowOpacity(win, 1.0f);
        SDL_SetWindowFullscreen(win, SDL_WINDOW_FULLSCREEN_DESKTOP);
    }
}

static void handle_action(view *v, cp_key_action a, double now) {
    cp_script *s = (cp_script *)v->s;
    switch (a) {
    case CPK_QUIT: v->running = 0; break;
    case CPK_PAUSE: cp_engine_toggle_pause(v->e);
        toast(v, now, "%s", v->p->paused ? "PAUSED" : "resumed"); break;
    case CPK_SPEED_UP:   cp_engine_set_speed(v->e, v->p->speed_wps * 1.1);
        toast(v, now, "%.1f w/s", v->p->speed_wps); break;
    case CPK_SPEED_DOWN: cp_engine_set_speed(v->e, v->p->speed_wps / 1.1);
        toast(v, now, "%.1f w/s", v->p->speed_wps); break;
    case CPK_SCROLL_FWD:  cp_engine_scroll(v->e, +2); break;
    case CPK_SCROLL_BACK: cp_engine_scroll(v->e, -2); break;
    case CPK_PAGE_NEXT: case CPK_PAGE_PREV: {
        int pg = cp_page_of(s, cp_pacer_word_index(v->p));
        int np = a == CPK_PAGE_NEXT ? pg + 1 : pg - 1;
        if (np >= 0 && np < (int)s->np) {
            cp_engine_jump(v->e, s->pages[np].w0, 1);
            toast(v, now, "page %d/%zu: %s", np + 1, s->np, s->pages[np].name);
        }
        break; }
    case CPK_MODE: cp_engine_cycle_mode(v->e);
        toast(v, now, "mode: %s",
              s->default_mode == CP_MODE_VOICE ? "voice" :
              s->default_mode == CP_MODE_TRACK ? "track" : "classic"); break;
    case CPK_RECAL: cp_engine_vad_recalibrate(v->e);
        toast(v, now, "mic recalibrating (stay quiet 0.5s)"); break;
    case CPK_SENS_UP:   cp_engine_sensitivity(v->e, +0.5); toast(v, now, "sens +"); break;
    case CPK_SENS_DOWN: cp_engine_sensitivity(v->e, -0.5); toast(v, now, "sens -"); break;
    case CPK_JUMP_HOME: cp_engine_jump(v->e, 0, 1); toast(v, now, "restart"); break;
    case CPK_MIRROR: v->mirror = (v->mirror + 1) % 4;
        toast(v, now, "mirror: %s",
              v->mirror==0?"off":v->mirror==1?"horizontal":v->mirror==2?"vertical":"180°"); break;
    case CPK_OVERLAY: v->style = v->style == CP_OVERLAY_FULLSCREEN
                                ? CP_OVERLAY_NOTCH : CP_OVERLAY_FULLSCREEN;
        toast(v, now, "overlay: %s", v->style == CP_OVERLAY_NOTCH ? "notch" : "fullscreen");
        break;
    case CPK_HELP: toast(v, now, "%s", "see console for key list"); break;
    default: break;
    }
}

#ifdef CP_HAVE_SDL_TTF
static void draw_text(view *v, SDL_Renderer *ren, int x, int y, const char *utf8,
                      SDL_Color c, int small) {
    SDL_Surface *su = TTF_RenderUTF8_Blended(small ? v->font_small : v->font, utf8, c);
    if (!su) return;
    SDL_Texture *tx = SDL_CreateTextureFromSurface(ren, su);
    int flip = 0; SDL_RendererFlip f = SDL_FLIP_NONE;
    if (v->mirror == 1 || v->mirror == 3) { f = SDL_FLIP_HORIZONTAL; flip = 1; }
    if (v->mirror == 2 || v->mirror == 3) { /* vertical: render copy flipped too */
        f = flip ? (SDL_RendererFlip)(f | SDL_FLIP_VERTICAL) : SDL_FLIP_VERTICAL; }
    SDL_Rect dst = { x, y, su->w, su->h };
    SDL_RenderCopyEx(ren, tx, NULL, &dst, 0, NULL, f);
    SDL_DestroyTexture(tx); SDL_FreeSurface(su);
}
#endif

static Uint8 RGBx(Uint32 c, int shift) { return (Uint8)((c >> shift) & 0xFF); }

int cp_overlay_sdl_run(cp_engine *e, int loop, cp_overlay_style style) {
    const cp_script *s = (const cp_script *)e->s;  /* engine keeps script ptr */
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }
    view v = { .e = e, .s = s, .p = cp_engine_pacer(e), .style = style };
#ifdef CP_HAVE_SDL_TTF
    TTF_Init();
    v.font = TTF_OpenFont("assets/fonts.ttf", s->font_size > 0 ? s->font_size : 28);
    v.font_small = TTF_OpenFont("assets/fonts.ttf", 14);
#endif
    SDL_Window *win = SDL_CreateWindow("cprompter",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 960, 540,
        SDL_WINDOW_ALLOW_HIGHDPI | (style == CP_OVERLAY_FULLSCREEN
                                    ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0));
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1,
                            SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!ren) ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);

    cp_engine_start(e);
    v.running = 1;
    double last = cp_engine_now();
    while (v.running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            double now = cp_engine_now();
            if (ev.type == SDL_QUIT) v.running = 0;
            if (ev.type == SDL_KEYDOWN) {
                int ch, ctrl, alt;
                sdl_key_to_logical(&ev.key, &ch, &ctrl, &alt);
                cp_key_action a = cpk_lookup(ch, ctrl, alt);
                if (a == CPK_OVERLAY) { handle_action(&v, a, now); apply_style(win, v.style); }
                else handle_action(&v, a, now);
            }
            if (ev.type == SDL_MOUSEWHEEL)
                cp_engine_scroll(e, ev.wheel.preciseY * 1.0);
        }

        double now = cp_engine_now(), dt = now - last; last = now;
        if (dt > 0.1) dt = 0.1;
        /* pacer tick happens here (UI owns the clock) */
        cp_pacer_tick((cp_pacer *)v.p, now, dt);

        if (cp_pacer_finished(v.p)) {
            if (loop) cp_pacer_init((cp_pacer *)v.p, (cp_script *)s, v.p->speed_wps);
            else v.running = 0;
        }

        /* ---- render ---- */
        int W, H; SDL_GetRendererOutputSize(ren, &W, &H);
        SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(ren, 0, 0, 0, v.style == CP_OVERLAY_NOTCH ? 200 : 255);
        SDL_RenderClear(ren);

        size_t cur = cp_pacer_word_index(v.p);
        int line_h = v.style == CP_OVERLAY_NOTCH ? 34 : 44;
        int center_y = v.style == CP_OVERLAY_NOTCH ? 10 : H * 3 / 10;
        Uint32 hlc = hl_colors[s->highlight_color % 6];

#ifdef CP_HAVE_SDL_TTF
        if (v.font) {
            int x = 60, y = center_y;
            int maxw = W - 120;
            for (size_t i = (cur > 6 ? cur - 6 : 0);
                 i < s->nw && i < cur + 20 && y < H - line_h * 2; i++) {
                SDL_Color c = (i == cur)
                    ? (SDL_Color){ RGBx(hlc,16), RGBx(hlc,8), RGBx(hlc,0), 255 }
                    : (i < cur ? (SDL_Color){90,90,90,255} : (SDL_Color){220,220,220,255});
                int tw = 0; TTF_SizeUTF8(v.font, s->words[i].text, &tw, NULL);
                if (x + tw + 12 > 60 + maxw) { x = 60; y += line_h; }
                draw_text(&v, ren, x, y, s->words[i].text, c, 0);
                x += tw + 12;
            }
            /* status strip: mode | speed | wpm | mic bar | page */
            char st[128];
            snprintf(st, sizeof st, "%s  %.1f w/s  %.0f wpm  page %d/%zu%s",
                     s->default_mode == CP_MODE_VOICE ? "VOICE" :
                     s->default_mode == CP_MODE_TRACK ? "TRACK" : "CLASSIC",
                     v.p->speed_wps, cp_engine_wpm(e),
                     cp_page_of(s, cur) + 1, s->np,
                     v.p->paused ? "  [PAUSED]" : "");
            SDL_Color gc = { 150, 150, 150, 255 };
            draw_text(&v, ren, 60, H - 26, st, gc, 1);
            float lvl = cp_engine_level(e);
            SDL_SetRenderDrawColor(ren, cp_engine_speaking(e) ? 0 : 60,
                                   cp_engine_speaking(e) ? 200 : 60, 80, 255);
            SDL_Rect mb = { W - 180, H - 22, (int)(lvl * 120), 8 };
            SDL_RenderFillRect(ren, &mb);
            if (now < v.toast_until) {
                SDL_Color tc = { 255, 220, 120, 255 };
                draw_text(&v, ren, W / 2 - 120, center_y - line_h - 24,
                          v.toast, tc, 1);
            }
        } else
#endif
        {
            /* block-meter fallback (no SDL_ttf): still shows cursor+progress */
            int n = (int)s->nw > 200 ? 200 : (int)s->nw;
            if (n < 1) n = 1;
            for (int i = 0; i < n; i++) {
                int on = (int)cur * n / (int)(s->nw ? s->nw : 1) >= i;
                SDL_SetRenderDrawColor(ren, on?120:40, on?120:40, on?120:40, 255);
                SDL_Rect r = { 40 + i * (W - 80) / n, center_y, (W - 80)/n - 2, 10 };
                SDL_RenderFillRect(ren, &r);
            }
            SDL_SetRenderDrawColor(ren, RGBx(hlc,16), RGBx(hlc,8), RGBx(hlc,0), 255);
            SDL_Rect cr = { 40 + (int)cur * (W - 80) / n, center_y - 4, 4, 18 };
            SDL_RenderFillRect(ren, &cr);
        }
        SDL_RenderPresent(ren);
        SDL_Delay(8);
    }
    cp_engine_stop(e);
#ifdef CP_HAVE_SDL_TTF
    if (v.font) TTF_CloseFont(v.font);
    if (v.font_small) TTF_CloseFont(v.font_small);
    TTF_Quit();
#endif
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
