/* overlay_sdl.c — M0/M1 SDL2 renderer: fullscreen black panel, word list
 * scrolling upward, current word highlighted in the chosen color. Keys:
 *   Space = pause/resume   Esc/Q = quit   Up/Down or wheel = manual scroll
 *   +/- = speed   T/V/C = track/voice/classic mode toggle (page-level)
 * Mirror flip (H/V/180) via SDL_RenderGeometry-free approach: we just draw
 * text with flipped layout flags for now; true GPU flip lands with M2 fonts.
 * NOTE: this build uses SDL_ttf if available; otherwise falls back to a
 * coarse "block meter" render so the loop still exercises on any machine. */
#include "overlay_sdl.h"

#include <SDL.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef CP_HAVE_SDL_TTF
#include <SDL_ttf.h>
#endif

static const Uint32 hl_colors[6] = {
    0xFFFFFFFF, /* white   */
    0xFFFFFF00, /* yellow  */
    0xFF00FF80, /* green   */
    0xFF40A0FF, /* blue    */
    0xFFFF69B4, /* pink    */
    0xFFFF8C00  /* orange  */
};

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int cp_overlay_sdl_run(const cp_script *s, cp_pacer *p, int loop) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }
#ifdef CP_HAVE_SDL_TTF
    TTF_Init();
    TTF_Font *font = TTF_OpenFont("assets/fonts.ttf", s->font_size > 0 ? s->font_size : 28);
#endif
    SDL_Window *win = SDL_CreateWindow("cprompter",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 960, 540,
        SDL_WINDOW_FULLSCREEN_DESKTOP | SDL_WINDOW_BORDERLESS | SDL_WINDOW_ALLOW_HIGHDPI);
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!ren) ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);

    int running = 1;
    double last = now_sec();
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = 0;
            if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                case SDLK_ESCAPE: case SDLK_q: running = 0; break;
                case SDLK_SPACE: cp_pacer_set_paused(p, !p->paused); break;
                case SDLK_UP:    cp_pacer_scroll_by(p, +2); break;
                case SDLK_DOWN:  cp_pacer_scroll_by(p, -2); break;
                case SDLK_PLUS: case SDLK_EQUALS: p->speed_wps *= 1.1; break;
                case SDLK_MINUS: p->speed_wps /= 1.1; break;
                case SDLK_c: ((cp_script*)s)->default_mode = CP_MODE_CLASSIC; break;
                case SDLK_v: ((cp_script*)s)->default_mode = CP_MODE_VOICE;   break;
                case SDLK_t: ((cp_script*)s)->default_mode = CP_MODE_TRACK;   break;
                default: break;
                }
            }
            if (e.type == SDL_MOUSEWHEEL)
                cp_pacer_scroll_by(p, e.wheel.preciseY * 1.0);
        }

        double now = now_sec(), dt = now - last; last = now;
        if (dt > 0.1) dt = 0.1;
        cp_pacer_tick(p, now, dt);

        if (cp_pacer_finished(p)) {
            if (loop) cp_pacer_init(p, s, p->speed_wps);
            else running = 0;
        }

        /* ---- render ---- */
        SDL_SetRenderDrawColor(ren, 0, 0, 0, SDL_ALPHA_OPAQUE);
        SDL_RenderClear(ren);
        int W, H; SDL_GetRendererOutputSize(ren, &W, &H);
        int line_h = 44;
        int center_y = H * 3 / 10;               /* read line near top third */

#ifdef CP_HAVE_SDL_TTF
        if (font) {
            size_t cur = cp_pacer_word_index(p);
            /* draw words from cur-8 .. cur+24, wrapping lines; highlight cur */
            int x = 60, y = center_y - line_h * 2;
            int maxw = W - 120;
            for (size_t i = (cur > 8 ? cur - 8 : 0); i < s->nw && i < cur + 24 && y < H - line_h; i++) {
                SDL_Color c = (i == cur) ? (SDL_Color){
                    (hl_colors[s->highlight_color]>>24)&0xFF,
                    (hl_colors[s->highlight_color]>>16)&0xFF,
                    (hl_colors[s->highlight_color]>>8)&0xFF, 255 }
                    : (i < cur ? (SDL_Color){90,90,90,255} : (SDL_Color){220,220,220,255});
                SDL_Surface *su = TTF_RenderUTF8_Blended(font, s->words[i].text, c);
                SDL_Texture *tx = SDL_CreateTextureFromSurface(ren, su);
                int tw = su->w, th = su->h;
                if (x + tw + 12 > 60 + maxw) { x = 60; y += line_h; }
                SDL_Rect dst = { x, y, tw, th };
                SDL_RenderCopy(ren, tx, NULL, &dst);
                x += tw + 12;
                SDL_DestroyTexture(tx); SDL_FreeSurface(su);
            }
            /* read line marker */
            SDL_SetRenderDrawColor(ren, 60, 60, 60, 255);
            SDL_Rect lr = { 0, center_y - line_h * 2 - 6, W, 1 };
            SDL_RenderFillRect(ren, &lr);
        }
#else
        /* fallback block meter: one small rect per word, cursor bright */
        size_t cur = cp_pacer_word_index(p);
        SDL_SetRenderDrawColor(ren, 255, 255, 255, 255);
        int n = (int)s->nw > 200 ? 200 : (int)s->nw;
        for (int i = 0; i < n; i++) {
            int on = (i <= (int)cur);
            SDL_SetRenderDrawColor(ren, on?120:40, on?120:40, on?120:40, 255);
            SDL_Rect r = { 40 + i * (W - 80) / n, center_y, (W - 80)/n - 2, 10 };
            SDL_RenderFillRect(ren, &r);
        }
        SDL_SetRenderDrawColor(ren, (hl_colors[s->highlight_color]>>24)&0xFF,
                               (hl_colors[s->highlight_color]>>16)&0xFF,
                               (hl_colors[s->highlight_color]>>8)&0xFF, 255);
        SDL_Rect cr = { 40 + (int)cur * (W - 80) / n, center_y - 4, 4, 18 };
        SDL_RenderFillRect(ren, &cr);
#endif
        SDL_RenderPresent(ren);
        SDL_Delay(8);
    }
#ifdef CP_HAVE_SDL_TTF
    if (font) TTF_CloseFont(font);
    TTF_Quit();
#endif
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
