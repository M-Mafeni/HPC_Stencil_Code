stencil: stencil.c
	icc -fast  -std=c99 -pg -Wall $^ -o $@

