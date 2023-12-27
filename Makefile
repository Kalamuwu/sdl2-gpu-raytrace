CUDA_PATH     ?= /opt/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# select one of these for Debug vs. Release
#NVCC_DBG       = -g
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64 --library SDL2,SDL2_image -forward-unknown-to-host-compiler -O3 -ffast-math -Wall -ftz=true -use_fast_math
GENCODE_FLAGS  = -gencode arch=compute_86,code=sm_86

# compile main.cu
cudart:
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o build/cudaRT.o -c main.cu    # link
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o build/cudaRT build/cudaRT.o  # compile

# profiler
profile_basic: cudaRT
	ncu build/cudaRT

# use nvprof --query-metrics
profile_metrics: cudaRT
	ncu --metrics smsp__inst_executed_pipe_fma,smsp__inst_executed_pipe_fp64 build/cudaRT

clean:
	rm -f build/*
