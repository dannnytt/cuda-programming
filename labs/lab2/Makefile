CXX = g++
NVCC = nvcc

CXXFLAGS = -Wall
PTHREAD_FLAGS = -pthread 
NVCC_FLAGS = -arch=sm_75

BIN_DIR = bin
SRC_DIR = src

SOURCES_CPP = $(SRC_DIR)/first_second_task.cpp
SOURCES_CUDA = $(SRC_DIR)/first_task.cu $(SRC_DIR)/third_task.cu

EXECUTABLES_CPP = $(patsubst $(SRC_DIR)/%.cpp, $(BIN_DIR)/%, $(SOURCES_CPP))
EXECUTABLES_CUDA = $(patsubst $(SRC_DIR)/%.cu, $(BIN_DIR)/%, $(SOURCES_CUDA))

first_second_task: $(BIN_DIR)/first_second_task
first_task: $(BIN_DIR)/first_task
third_task: $(BIN_DIR)/third_task


all: $(BIN_DIR) $(EXECUTABLES_CPP) $(EXECUTABLES_CUDA)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)
	
$(BIN_DIR)/first_second_task: $(SRC_DIR)/first_second_task.cpp
	$(CXX) $(PTHREAD_FLAGS) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/first_task: $(SRC_DIR)/first_task.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BIN_DIR)/third_task: $(SRC_DIR)/third_task.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -rf $(BIN_DIR)/*
