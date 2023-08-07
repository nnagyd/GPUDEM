CompilerFlags = -O3 -std=c++17 -lineinfo -maxrregcount=48 --ptxas-options=-v --use_fast_math --gpu-architecture=sm_86
#CompilerFlags = -std=c++11 --gpu-architecture=sm_86

all: gpudem

gpudem:
	@echo -------------------------- Compiling GPUDEM Solver -------------------------- 
	nvcc -o GPUDEM $(CompilerFlags) main.cu
	@echo -------------------------------- GPUDEM ready ------------------------------- 

ex1:
	@echo ---------------------------- Compiling Example 1 ---------------------------- 
	nvcc -o GPUDEM_EX1 $(CompilerFlags) ex1_deposition.cu
	@echo -------------------------------- GPUDEM ready ------------------------------- 

ex2:
	@echo ---------------------------- Compiling Example 2 ---------------------------- 
	nvcc -o GPUDEM_EX2 $(CompilerFlags) ex2_layered_deposition.cu
	@echo -------------------------------- GPUDEM ready ------------------------------- 

ex3:
	@echo ---------------------------- Compiling Example 3 ---------------------------- 
	nvcc -o GPUDEM_EX3 $(CompilerFlags) ex3_deposition2.cu
	@echo -------------------------------- GPUDEM ready ------------------------------- 

clean:
	@echo ------------------------- Clening up GPUDEM Solver -------------------------- 
	rm GPUDEM
	rm GPUDEM_EX1
	rm GPUDEM_EX2
	rm GPUDEM_EX3
	@echo ------------------------------- Cleanup ready ------------------------------- 