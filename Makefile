CC_PATH      ?= /opt/rocm
CC          = $(CC_PATH)/bin/hipcc -v

BUILD_DIR     ?= build

EXECUTABLE     = hipRT
LIBRARIES      = -lSDL2 -lSDL2_image
INCLUDES       = -I/opt/rocm/include -Iinclude

# select one of these for Debug vs. Release
#CC_DBG        = -g
CC_DBG        =

CCFLAGS       = $(CC_DBG) $(INCLUDES) -std=c++11 -m64 -O3 -ffast-math -Wall -Wextra -Wpedantic
LDFLAGS       = $(CC_DBG) $(LIBRARIES)

SOURCES = screen.cpp main.cu
OBJECTS_C = $(SOURCES:.c=.o)
OBJECTS_C_CPP = $(OBJECTS_C:.cpp=.o)
OBJECTS_C_CPP_CU = $(OBJECTS_C_CPP:.cu=.o)
OBJECTS = $(addprefix ${BUILD_DIR}/,$(OBJECTS_C_CPP_CU))

.PHONY: all
.DEFAULT: all

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $(BUILD_DIR)/$(EXECUTABLE) $(OBJECTS)

$(BUILD_DIR)/%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@
$(BUILD_DIR)/%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@
$(BUILD_DIR)/%.o: %.cu
	$(CC) $(CCFLAGS) -c $< -o $@

# # profiler
# profile_basic: hipRT
# 	ncu $(BUILD_DIR)/hipRT
#
# # use nvprof --query-metrics
# profile_metrics: hipRT
# 	ncu --metrics smsp__inst_executed_pipe_fma,smsp__inst_executed_pipe_fp64 $(BUILD_DIR)/cudaRT

clean:
	rm -f $(BUILD_DIR)/*
