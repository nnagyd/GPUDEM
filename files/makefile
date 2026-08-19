
.PHONY: all examples clean e1 e2

CUDA_STD ?= c++14
EXTRA_INCLUDE ?=
CompilerFlags ?= -O3 -std=$(CUDA_STD) --allow-unsupported-compiler -lineinfo -maxrregcount=32 --ptxas-options=-v --use_fast_math --gpu-architecture=sm_86 -I../ $(EXTRA_INCLUDE)
LinkFlags ?= -lstdc++fs
all: examples

examples: e1 e2

e1:
	@echo -------------------------- Compiling GPUDEM Solver -------------------------- 
	nvcc -o GPUDEM $(CompilerFlags) e1_deposition.cu $(LinkFlags)
	@echo -------------------------------- GPUDEM ready ------------------------------- 

e2:
	@echo -------------------------- Compiling GPUDEM Solver -------------------------- 
	nvcc -o GPUDEM $(CompilerFlags) e2_CPT.cu $(LinkFlags)
	@echo -------------------------------- GPUDEM ready ------------------------------- 

clean:
	@echo ------------------------- Clening up GPUDEM Solver -------------------------- 
	@rm -f GPUDEM
	@rm -rf output
	@echo ------------------------------- Cleanup ready ------------------------------- 
