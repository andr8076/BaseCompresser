CC ?= cc
CFLAGS ?= -O3 -march=native -flto -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?= -flto
LDLIBS ?= -lm -pthread -ldl

BIN := basecompresser
PORTABLE_BIN := basecompresser-portable
SRC := src/basecompresser.c
DEPS := $(wildcard src/*.inc)

.PHONY: all clean test test-gpu test-portable portable bench
all: $(BIN)

$(BIN): $(SRC) $(DEPS)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(BIN) $(PORTABLE_BIN)

test: $(BIN)
	bash tests/test.sh

test-gpu: $(BIN)
	bash tests/test_gpu_equivalence.sh

$(PORTABLE_BIN): $(SRC) $(DEPS)
	$(CC) $(filter-out -march=native,$(CFLAGS)) $(SRC) -o $@ $(LDFLAGS) $(LDLIBS)

portable: $(PORTABLE_BIN)

test-portable: $(PORTABLE_BIN)
	BASECOMPRESSER_BIN=./$(PORTABLE_BIN) bash tests/test.sh

bench: $(BIN)
	python3 bench/benchmark.py
