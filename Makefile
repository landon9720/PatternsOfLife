CC=clang
CXX=clang++
CPPFLAGS=-std=c++1y -g -I/usr/local/include -O3

.PHONY: clean buddies.html

all: buddies

buddies: buddies.o easygame.o
	clang++ -O3 -o buddies buddies.o easygame.o -L/usr/local/lib -lSDL2 -lSDL2_image -lSDL2_mixer -framework OpenGL -lfann -lBox2D

buddies.html:
	emcc -DDISABLE_PARALLEL_FANN -s LEGACY_GL_EMULATION=1 -Ifann/include --preload-file assets/pickup.wav -o buddies.html fann/floatfann.c buddies.cpp easygame_emscripten.cpp

clean:
	rm -f *.o buddies buddies.js buddies.html
