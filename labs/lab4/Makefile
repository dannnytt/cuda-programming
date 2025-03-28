NVCC = nvcc
FLAGS = -arch=sm_75

BIN_DIR = bin
SRC_DIR = src

CUDA_SOURCES = $(wildcard $(SRC_DIR)/*.cu)
TARGETS = $(notdir $(basename $(CUDA_SOURCES)))
EXECUTABLES = $(addprefix $(BIN_DIR)/, $(TARGETS))

all: $(EXECUTABLES)

$(BIN_DIR)/%: $(SRC_DIR)/%.cu
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(FLAGS) $< -o $@

clean:
	rm -rf $(BIN_DIR)/*

.PHONY: all clean