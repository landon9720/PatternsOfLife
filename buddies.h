#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <set>
#include <fann.h>
#ifdef __EMSCRIPTEN__
#include "easygame_emscripten.h"
#include <emscripten.h>
#else
#include "easygame.h"
#endif

#include <OpenGL/gl.h>

using std::min;
using std::max;

#include <locale.h>
