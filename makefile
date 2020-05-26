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

.PHONY: dirs
dirs:
	@mkdir -p $(OBJDIR)
	@mkdir -p $(BINDIR)

.PHONY: build
build: dirs $(TARGET)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp 
	@echo "Compiling objects: $@"
	$(CC) $(CFLAGS) -MP -MMD -c $< -o $@ $(LDFLAGS)

$(TARGET): $(OBJECTS)
	@echo "Linking: $@"
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)
