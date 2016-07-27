CXX=clang++
CPPFLAGS=-std=c++1y -g -I/usr/local/include -O3

all: patterns

patterns: patterns.o easygame.o
	clang++ -O3 -o patterns patterns.o easygame.o -L/usr/local/lib -lSDL2 -lSDL2_image -lconfig++ -framework OpenGL

clean:
	rm -f *.o patterns
