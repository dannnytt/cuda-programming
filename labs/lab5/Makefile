# Компилятор и флаги
NVCC = nvcc
NVCCFLAGS = -arch=sm_75 -O2
PROFFLAGS = -lineinfo -Xcompiler -rdynamic

# Директории
SRC_DIR = src
BIN_DIR = bin
BUILD_DIR = build

# Файлы
SOURCES = $(wildcard $(SRC_DIR)/*.cu)
OBJECTS = $(patsubst $(SRC_DIR)/%.cu, $(BIN_DIR)/%.o, $(SOURCES))
EXECUTABLE = $(BUILD_DIR)/matrix_transpose
PROF_EXECUTABLE = $(BUILD_DIR)/matrix_transpose_prof

# Создание директорий
$(shell mkdir -p $(BIN_DIR) $(BUILD_DIR))

# Обычная сборка
all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(BIN_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Сборка с флагами профилирования
prof: $(PROF_EXECUTABLE)

$(PROF_EXECUTABLE): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(PROFFLAGS) -o $@ $^

# Очистка
clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR)
