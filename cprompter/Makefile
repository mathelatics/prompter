# cPrompter — pure-C teleprompter ("Textream for every device")
# Plan: docs/audit/cprompter-01-plan.md   Deps: docs/audit/cprompter-02-deps.md
#
# M0 core + M1 voice + M2 track plumbing: zero external deps by default.
# Optional builds:
#   make SDL=1        -> overlay renderer (needs libsdl2; pkg-config autodetected,
#                        SDL_ttf autodetected too if present)
#   make WHISPER=1    -> word-tracking ASR backend (needs third_party/whisper.cpp
#                        built with `cmake -B build && cmake --build build`)
#   make SERVER=1     -> M3 remote viewer (:7373) + director mode (:7575)
#                        (pure POSIX sockets, no civetweb needed)
CC      ?= cc
CFLAGS  ?= -O2 -std=c11 -Wall -Wextra -Wno-unused-parameter -g
LDFLAGS ?=
LDLIBS  ?= -lm -lpthread -ldl

CORE_SRC := main.c \
            core/script.c core/pacer.c core/engine.c \
            format/script_io.c format/pptx.c \
            audio/audio.c audio/vad.c audio/wpm.c \
            audio/align.c audio/track.c audio/wav.c \
            ui/hotkeys.c
SERVER_SRC := server/http_ws.c server/remote.c vendor/qrcodegen.c

SRCDIRS := core format audio ui
SRC     := $(CORE_SRC) $(if $(filter 1,$(SERVER)), $(SERVER_SRC))
OBJ     := $(SRC:.c=.o)
BIN     := cprompter

ifeq ($(SDL),1)
  SRC += ui/overlay_sdl.c
  OBJ := $(SRC:.c=.o)
  CFLAGS += -DCP_HAVE_SDL=1 $(shell pkg-config --cflags sdl2 2>/dev/null || echo -I/usr/include/SDL2)
  LDLIBS += $(shell pkg-config --libs sdl2 2>/dev/null || echo -lSDL2)
  ifeq ($(shell pkg-config --exists SDL2_ttf 2>/dev/null && echo yes),yes)
    CFLAGS += -DCP_HAVE_SDL_TTF=1 $(shell pkg-config --cflags SDL2_ttf)
    LDLIBS += $(shell pkg-config --libs SDL2_ttf)
  endif
endif

ifeq ($(WHISPER),1)
  WROOT ?= third_party/whisper.cpp
  CFLAGS += -DCP_HAVE_WHISPER=1 -I$(WROOT)/include -I$(WROOT)/src
  LDLIBS += -L$(WROOT)/build/src -lwhisper -lggml-base $(shell [ -f $(WROOT)/build/src/libggml-cpu.a ] && echo $(WROOT)/build/src/libggml-cpu.a)
endif

TEST_SRC := tools/test_core.c \
            core/script.c core/pacer.c core/engine.c \
            format/script_io.c format/pptx.c \
            audio/audio.c audio/vad.c audio/wpm.c \
            audio/align.c audio/track.c audio/wav.c \
            ui/hotkeys.c
TEST_BIN := cp_tests

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_SRC)
	$(CC) $(CFLAGS) -DCP_TESTING=1 -o $@ $(TEST_SRC) -lm -lpthread -ldl

demo: $(BIN)
	./$(BIN) samples/demo.txt --speed 4

clean:
	rm -f $(OBJ) $(BIN) $(TEST_BIN) /tmp/cp_*.textream /tmp/cp_*.wav

.PHONY: all test demo clean
