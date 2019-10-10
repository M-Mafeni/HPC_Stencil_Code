stencil: stencil.c
	gcc -Ofast -mtune=native -std=c99 -pg -Wall $^ -o $@

