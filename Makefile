CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c99
AR ?= ar

all: libhelix_codec.a hxq_demo

libhelix_codec.a: helix_codec.o
	$(AR) rcs $@ $<

helix_codec.o: helix_codec.c helix_codec.h
	$(CC) $(CFLAGS) -c -o $@ $<

hxq_demo: example.c libhelix_codec.a
	$(CC) $(CFLAGS) -o $@ $< -L. -lhelix_codec -lm

clean:
	rm -f *.o *.a hxq_demo

.PHONY: all clean
