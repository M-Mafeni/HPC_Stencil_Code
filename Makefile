stencil: stencil.c
	gcc -Ofast  -std=c99 -pg -Wall $^ -o $@

