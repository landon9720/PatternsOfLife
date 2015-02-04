CC=clang
CXX=clang++
CPPFLAGS=-std=c++1y -g

.PHONY: clean

all: buddies

buddies: buddies.o easygame.o
	clang++ -o buddies buddies.o easygame.o -lSDL2 -lSDL2_image -framework OpenGL -lfann

clean:
	rm *.o
