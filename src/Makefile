CC=gcc
CFLAGS=-I. -I./mylibs -I./extlibs -I/usr/include/SDL2 -O3 -Wall -Wextra -Werror  -Wno-unused-variable -Wno-unused-parameter -fopenmp 
LDFLAGS=-lSDL2  -lSDL2_ttf -lm  -lcudart
DEPS = $(wildcard *.h) $(wildcard mylibs/*.h) $(wildcard extlibs/*.h)
SRC_DIRS = $(wildcard mylibs/*.c) $(wildcard extlibs/*.c)
SRC_ROOT = $(wildcard *.c)
OBJ_DIRS = $(patsubst %.c,%.o,$(SRC_DIRS))
OBJ_ROOT = $(patsubst %.c,%.o,$(SRC_ROOT))

all: main run

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: $(OBJ_DIRS) $(OBJ_ROOT)
	@mkdir -p ../bin
	@printf "[compiling..."; $(CC) -o ../bin/main $^ $(CFLAGS) $(LDFLAGS); printf "\e[38;5;82mOK\e0]\n"

run: main
	@../bin/main

clean:
	rm -f *.o main mylibs/*.o extlibs/*.o ../bin/main