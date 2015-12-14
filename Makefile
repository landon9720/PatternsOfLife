CXX=clang++
CPPFLAGS=-std=c++1y -g -I/usr/local/include -O3

all: buddies

buddies: buddies.o easygame.o
	clang++ -O3 -o buddies buddies.o easygame.o -L/usr/local/lib -lSDL2 -lSDL2_image -framework OpenGL

clean:
	rm -f *.o buddies
