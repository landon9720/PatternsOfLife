CC=clang
CXX=clang++
CPPFLAGS=-std=c++1y -g -I/usr/local/include

.PHONY: clean

all: buddies

buddies: buddies.o easygame.o
	clang++ -o buddies buddies.o easygame.o -L/usr/local/lib -lSDL2 -lSDL2_image -framework OpenGL -lfann

clean:
	rm *.o
