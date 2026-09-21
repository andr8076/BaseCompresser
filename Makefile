CC ?= cc
CFLAGS ?= -O3 -march=native -flto -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?= -flto
LDLIBS ?= -lm -pthread -ldl

BIN := basecompresser
SRC := src/basecompresser.c
DEPS := $(wildcard src/*.inc)

.PHONY: all clean test test-gpu
all: $(BIN)

$(BIN): $(SRC) $(DEPS)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(BIN)

test: $(BIN)
	bash tests/test.sh

test-gpu: $(BIN)
	bash tests/test_gpu_equivalence.sh
