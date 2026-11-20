NVCC      ?= nvcc
# Set ARCH for your GPU, e.g. `make ARCH=-arch=sm_75` for a T4
ARCH      ?=
NVCCFLAGS ?= -O3 -std=c++17 $(ARCH) -Iinclude
LDLIBS    := -lcublas

BIN_DIR   := bin
KERNELS   := $(wildcard src/kernels/*.cu)
HEADERS   := $(wildcard include/*.cuh)

MAIN      := $(BIN_DIR)/matmul
TEST      := $(BIN_DIR)/test_matmul

.PHONY: all run test clean

all: $(MAIN) $(TEST)

$(MAIN): src/main.cu src/benchmark.cu $(KERNELS) $(HEADERS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) src/main.cu src/benchmark.cu $(KERNELS) -o $@ $(LDLIBS)

$(TEST): tests/test_matmul.cu $(KERNELS) $(HEADERS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) tests/test_matmul.cu $(KERNELS) -o $@ $(LDLIBS)

$(BIN_DIR):
	mkdir -p $@

run: $(MAIN)
	./$(MAIN)

test: $(TEST)
	./$(TEST)

clean:
	rm -rf $(BIN_DIR)
