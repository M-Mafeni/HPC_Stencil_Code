stencil: stencil.c
	mpiicc -O3 -xHost -std=c99 -Wall $^ -o $@

