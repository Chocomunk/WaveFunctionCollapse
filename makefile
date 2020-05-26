# C++ Compiler
CC = g++
CFLAGS = -g -Wall
OPENCV = opencv4
LDFLAGS = `pkg-config --libs --cflags $(OPENCV)`

# Folders 
BINDIR = bin
OBJDIR = obj
SRCDIR = cpp

# Files
SRC = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SRC:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
TARGET = $(BINDIR)/wfc


.PHONY: all
all: build

.PHONY: clean
clean:
	@echo "Cleaning ..."
	@rm -rf $(OBJDIR)
	@rm -rf $(BINDIR)
	@rm -rf results

.PHONY: dirs
dirs:
	@mkdir -p $(OBJDIR)
	@mkdir -p $(BINDIR)
	@mkdir -p results

.PHONY: build
build: dirs $(TARGET)

.PHONY: test
test:
	bin/wfc tiles/red/ 2 1 1 64 64 red.png 0
	bin/wfc tiles/spirals/ 3 1 1 64 64 spirals.png 0
	bin/wfc tiles/bricks/ 3 0 1 64 64 bricks.png 0
	bin/wfc tiles/dungeons/ 3 0 1 64 64 dungeons.png 0
	bin/wfc tiles/paths/ 3 0 1 64 64 paths.png 0

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp 
	@echo "Compiling objects: $@"
	$(CC) $(CFLAGS) -MP -MMD -c $< -o $@ $(LDFLAGS)

$(TARGET): $(OBJECTS)
	@echo "Linking: $@"
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)
