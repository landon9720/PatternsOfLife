#include <cassert>
#include <memory>
#include <SDL/SDL.h>
#include <SDL/SDL_mixer.h>
#include <gl/gl.h>
#include "easygame_emscripten.h"

#define CHECK_INIT assert(initialized == true)

static bool initialized = false;
static SDL_Window *window = nullptr;
static const unsigned char *keystate = nullptr;

void eg_init(int width, int height, const std::string &title) {
  SDL_Init(SDL_INIT_VIDEO);
  SDL_SetVideoMode(width, height, 32, SDL_OPENGL|SDL_DOUBLEBUF);

  keystate = SDL_GetKeyboardState(nullptr);

  glMatrixMode(GL_PROJECTION);
  glOrtho(0, width, 0, height, -1, 1);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  Mix_Init(0);
  Mix_OpenAudio(MIX_DEFAULT_FREQUENCY, MIX_DEFAULT_FORMAT, 2, 1024);

  initialized = true;
}

void eg_shutdown() {
  CHECK_INIT;
  SDL_Quit();
}

bool eg_poll_event(EGEvent *ev) {
  CHECK_INIT;
  return SDL_PollEvent(ev);
}

bool eg_get_keystate(int scancode) {
  return keystate[scancode];
}

//void eg_push_transform();
//void eg_pop_transform();

void eg_reset_transform() {
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

//void eg_rotate(float r);
//void eg_scale(float x, float y);

void eg_translate(float x, float y) {
  glMatrixMode(GL_MODELVIEW);
  glTranslatef(x, y, 0.0f);
}

void eg_swap_buffers() {
  CHECK_INIT;
  SDL_GL_SwapWindow(window);
}

void eg_clear_screen(float r, float g, float b, float a) {
  CHECK_INIT;
  glClearColor(r, g, b, a);
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
}

void eg_set_color(float r, float g, float b, float a) {
  glColor4f(r, g, b, a);
}

void eg_draw_line(float x0, float y0, float x1, float y1, float w) {
  glLineWidth(w);
  glBegin(GL_LINES);
    glVertex2f(x0, y0);
    glVertex2f(x1, y1);
  glEnd();
}

void eg_draw_square(float x, float y, float w, float h) {
  glBegin(GL_QUADS);
    glVertex2f(x, y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y + h);
    glVertex2f(x, y + h);
  glEnd();
}

/*
struct EGImage {
  GLuint tex;
};

EGImage *eg_load_image(const std::string &filename) {
  SDL_Surface *surface = IMG_Load(filename.c_str());

  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, surface->w, surface->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, surface->pixels);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  
  SDL_FreeSurface(surface);

  EGImage *image = new EGImage { texture };
  return image;
}

void eg_free_image(EGImage *image) {
  glDeleteTextures(1, &image->tex);
}

void eg_draw_image(EGImage *img, float x, float y, float w, float h) {
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, img->tex);

  glBegin(GL_QUADS);
    glTexCoord2f(0, 1); glVertex2f(x, y);
    glTexCoord2f(1, 1); glVertex2f(x+w, y);
    glTexCoord2f(1, 0); glVertex2f(x+w, y+h);
    glTexCoord2f(0, 0); glVertex2f(x, y+h);
  glEnd();

  glDisable(GL_TEXTURE);
  glBindTexture(GL_TEXTURE_2D, 0);
}
*/

struct EGSound {
  Mix_Chunk *chunk;
};

EGSound *eg_load_sound(const std::string &filename) {
  Mix_Chunk *chunk = Mix_LoadWAV(filename.c_str());
  assert(chunk != nullptr);
  EGSound *sound = new EGSound;
  sound->chunk = chunk;
  return sound;
}

void eg_free_sound(EGSound *sound) {
  Mix_FreeChunk(sound->chunk);
  delete sound;
}

void eg_play_sound(EGSound *sound) {
  Mix_PlayChannel(-1, sound->chunk, 0);
}
