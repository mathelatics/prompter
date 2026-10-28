# cPrompter — pure-C teleprompter ("Textream for every device")
# Plan: docs/audit/cprompter-01-plan.md   Deps: docs/audit/cprompter-02-deps.md
#
# M0 core: zero external deps (classic-mode CLI/headless + .textream IO).
# Optional builds:
#   make SDL=1        -> fullscreen/overlay renderer (needs libsdl2)
#   make WHISPER=1    -> word-tracking ASR (M2, needs third_party/whisper.cpp)
CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter -g
LDFLAGS ?=
LDLIBS  ?= -lm

SRCDIRS := core format
SRC     := main.c $(foreach d,$(SRCDIRS),$(wildcard $(d)/*.c))
OBJ     := $(SRC:.c=.o)
BIN     := cprompter

ifeq ($(SDL),1)
  SRC += ui/overlay_sdl.c
  OBJ := $(SRC:.c=.o)
  CFLAGS += -DCP_HAVE_SDL=1 $(shell pkg-config --cflags sdl2 2>/dev/null || echo -I/usr/include/SDL2)
  LDLIBS += $(shell pkg-config --libs sdl2 2>/dev/null || echo -lSDL2)
endif

TEST_SRC := tools/test_core.c core/script.c core/pacer.c format/script_io.c
TEST_BIN := cp_tests

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_SRC)
	$(CC) $(CFLAGS) -DCP_TESTING=1 -o $@ $(TEST_SRC) -lm

demo: $(BIN)
	./$(BIN) samples/demo.txt --speed 4

clean:
	rm -f $(OBJ) $(BIN) $(TEST_BIN) /tmp/cp_*.textream

.PHONY: all test demo clean
