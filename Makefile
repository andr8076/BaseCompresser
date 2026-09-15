CC ?= cc
CFLAGS ?= -O3 -march=native -flto -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?= -flto
LDLIBS ?= -lm -pthread

BIN := basecompresser
SRC := src/basecompresser.c

.PHONY: all clean test
all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(BIN)

test: $(BIN)
	bash tests/test.sh
