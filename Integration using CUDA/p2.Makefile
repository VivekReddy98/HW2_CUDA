
##vkarri Vivek Reddy Karri

compile:
	nvcc p2.cu -o p2 -O3 -lm -Wno-deprecated-gpu-targets -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64

run:
	./p2 200000

clean:
	rm p2
