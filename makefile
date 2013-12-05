all: macierz_cuda
	
macierz_cuda: main.cu
	nvcc main.cu -o macierz_cuda
	
clean:
	rm -f macierz_cuda
