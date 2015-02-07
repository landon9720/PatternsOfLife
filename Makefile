CC=clang
CXX=clang++
CPPFLAGS=-std=c++1y -g -I/usr/local/include

.PHONY: clean buddies.html

all: buddies

buddies: buddies.o easygame.o
	clang++ -o buddies buddies.o easygame.o -L/usr/local/lib -lSDL2 -lSDL2_image -framework OpenGL -lfann

buddies.html:
	emcc -DDISABLE_PARALLEL_FANN -s LEGACY_GL_EMULATION=1 -Ifann/include fann/floatfann.c buddies.cpp easygame_emscripten.cpp -o buddies.html

clean:
	rm -f *.o buddies buddies.js buddies.html
