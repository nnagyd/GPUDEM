
.PHONY: clean

CompilerFlags = -O3 -std=c++17 -lineinfo -maxrregcount=64 --ptxas-options=-v --use_fast_math --gpu-architecture=sm_86
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

ex4:
	@echo ---------------------------- Compiling Example 4 ---------------------------- 
	nvcc -o GPUDEM_EX4 $(CompilerFlags) ex4_multi_material.cu
	@echo -------------------------------- GPUDEM ready ------------------------------- 

ex5:
	@echo ---------------------------- Compiling Example 5 ---------------------------- 
	nvcc -o GPUDEM_EX5 $(CompilerFlags) ex5_STL_geometry.cu
	@echo -------------------------------- GPUDEM ready ------------------------------- 

ex6:
	@echo ---------------------------- Compiling Example 6 ---------------------------- 
	nvcc -o GPUDEM_EX6 $(CompilerFlags) ex6_validation.cu
	@echo -------------------------------- GPUDEM ready ------------------------------- 

ex7:
	@echo ---------------------------- Compiling Example 7 ---------------------------- 
	nvcc -o GPUDEM_EX7 $(CompilerFlags) ex7_movingSTL.cu
	@echo -------------------------------- GPUDEM ready ------------------------------- 

ex7_2:
	@echo ---------------------------- Compiling Example 7-2 ---------------------------- 
	nvcc -o GPUDEM_EX7_2 $(CompilerFlags) ex7_movingSTL_deposit.cu
	@echo -------------------------------- GPUDEM ready ------------------------------- 

ex8:
	@echo ---------------------------- Compiling Example 8 ---------------------------- 
	nvcc -o GPUDEM_EX8 $(CompilerFlags) ex8_movingTool.cu
	@echo -------------------------------- GPUDEM ready ------------------------------- 

ex8_2:
	@echo ---------------------------- Compiling Example 8_2 ---------------------------- 
	nvcc -o GPUDEM_EX8_2 $(CompilerFlags) ex8_movingTool_deposit.cu
	@echo -------------------------------- GPUDEM ready ------------------------------- 

clean:
	@echo ------------------------- Clening up GPUDEM Solver -------------------------- 
	@if [ -e GPUDEM ]; then \
		rm GPUDEM; \
	fi
	@if [ -e GPUDEM_EX1 ]; then \
		rm GPUDEM_EX1; \
	fi
	@if [ -e GPUDEM_EX2 ]; then \
		rm GPUDEM_EX2; \
	fi
	@if [ -e GPUDEM_EX3 ]; then \
		rm GPUDEM_EX3; \
	fi
	@if [ -e GPUDEM_EX4 ]; then \
		rm GPUDEM_EX4; \
	fi
	@if [ -e GPUDEM_EX5 ]; then \
		rm GPUDEM_EX5; \
	fi
	@if [ -e GPUDEM_EX6 ]; then \
		rm GPUDEM_EX6; \
	fi
	@if [ -e GPUDEM_EX7 ]; then \
		rm GPUDEM_EX7; \
	fi
	@if [ -e GPUDEM_EX7_2 ]; then \
		rm GPUDEM_EX7_2; \
	fi
	@if [ -e GPUDEM_EX8 ]; then \
		rm GPUDEM_EX8; \
	fi
	@if [ -e GPUDEM_EX8_2 ]; then \
		rm GPUDEM_EX8_2; \
	fi
	@echo ------------------------------- Cleanup ready ------------------------------- 