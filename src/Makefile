CC=gcc
NVCC=nvcc
NVCC_FLAGS=-O3
GCC_FLAGS=-I. -I./mylibs -I./extlibs -I/usr/include/SDL2 -O3 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -fopenmp 
GCC_LDFLAGS=-lSDL2  -lSDL2_ttf -lm -fopenmp
NVCC_LDFLAGS=-lcudart -lSDL2 -lSDL2_ttf -lm -Xcompiler -fopenmp
DEPS = $(wildcard *.h) $(wildcard mylibs/*.h) $(wildcard extlibs/*.h)
SRC_DIRS = $(wildcard mylibs/*.c) $(wildcard extlibs/*.c)
SRC_ROOT = $(wildcard *.c)
OBJ_DIRS = $(patsubst %.c,%.o,$(SRC_DIRS))
OBJ_ROOT = $(patsubst %.c,%.o,$(SRC_ROOT))
CU_SRC = $(wildcard *.cu)
CU_OBJ = $(patsubst %.cu,%.o,$(CU_SRC))

all: main run

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(GCC_FLAGS)

%.o: %.cu $(DEPS)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $< 

main: $(OBJ_DIRS) $(OBJ_ROOT) $(CU_OBJ)
	@mkdir -p ../bin
	@printf "[compiling..."; $(NVCC) -o ../bin/main $^ $(NVCC_LDFLAGS); printf "\e[38;5;82mOK\e0]\n"

run: main
	@../bin/main

clean:
	rm -f *.o main mylibs/*.o extlibs/*.o ../bin/main