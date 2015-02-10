#ifndef __EASYGAME_H_
#define __EASYGAME_H_

#include <string>
#include <SDL2/SDL.h>

// initialization, shutdown, etc.
void eg_init(int width, int height, const std::string &title);
void eg_shutdown();

typedef SDL_Event EGEvent;

bool eg_poll_event(EGEvent *ev);
bool eg_get_keystate(int scancode);

// graphics
void eg_swap_buffers();
void eg_clear_screen(float r, float g, float b, float a);
void eg_push_transform();
void eg_pop_transform();
void eg_reset_transform();
void eg_rotate(float r);
void eg_scale(float x, float y);
void eg_translate(float x, float y);

void eg_set_color(float r, float g, float b, float a);
void eg_draw_square(float x, float y, float w, float h);
void eg_draw_line(float x0, float y0, float x1, float y1, float w = 1.0f);

struct EGImage;
EGImage *eg_load_image(const std::string &filename);
void eg_free_image(EGImage *image);
void eg_draw_image(EGImage *img, float x, float y, float w, float h);

// audio
struct EGSound;
EGSound *eg_load_sound(const std::string &filename);
void eg_free_sound(EGSound *sound);
void eg_play_sound(EGSound *sound);

// util
template<typename T>
T clamp(const T &x, const T &a, const T &b) {
  return std::max(std::min(x, b), a);
}

#endif

