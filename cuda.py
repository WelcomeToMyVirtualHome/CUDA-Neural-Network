from numba import cuda
import numpy as np
#import cProfile, pstats, io
import time


@cuda.jit
def add_kernel(x, y, out):
    tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
    ty = cuda.blockIdx.x  # Similarly, this is the unique block ID within the 1D grid

    block_size = cuda.blockDim.x  # number of threads per block
    grid_size = cuda.gridDim.x    # number of blocks in the grid
    
    start = tx + ty * block_size
    stride = block_size * grid_size

    # assuming x and y inputs are same length
    for i in range(start, x.shape[0], stride):
        out[i] = x[i] + y[i]

@cuda.jit
def matrix_mul_cuda(A, B, out, N):
	row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
	col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
	tmpSum = 0;
	if row < N and col < N:
		for i in range(0,N):
			tmpSum += A[row * N + i] * B[i * N + col];
	out[row * N + col] = tmpSum

def matrix_mul(A, B, out, N):
	for row in range(0,N):
		tmpSum = 0
		for col in range(0,N):
			tmpSum += A[row * N + col] * B[row * N + col]

		out[row * N + col] = tmpSum
	return out

def vec_add_test():
	n = 100000
	x = np.arange(n).astype(np.float32)
	y = x
	out = np.empty_like(x)

	threads_per_block = 128
	blocks_per_grid = 30
	add_kernel[blocks_per_grid, threads_per_block](x, y, out)
	print(out[:20])

if __name__ == '__main__':
	
	N = 10
	A = np.identity(N).astype(np.float32)
	A = A.flatten()
	# B = np.ones(shape=(N,N)).astype(np.float32)
	# B = B.flatten()
	B = np.arange(N*N)
	C = np.zeros(shape=(N,N)).astype(np.float32)
	C = B.flatten()

	threads_per_block = N, N
	blocks_per_grid= 1, 1 
	if N * N > 1024:
		threads_per_block = 1024, 1024
		blocks_per_grid = int(np.ceil(N/threads_per_block[0])), int(np.ceil(N/threads_per_block[1]))
	print(blocks_per_grid)
	
	rep = 1000
	t0 = time.time()
	for i in range(0,rep):
		matrix_mul_cuda[blocks_per_grid,threads_per_block](A,B,C,N)

	t1 = time.time()
	total_n = t1-t0
	print('GPU matrix mul time {:f}s'.format(total_n))
	
	t0 = time.time()
	for i in range(0,rep):
		C = matrix_mul(A,B,C,N)

	print(C)
	t1 = time.time()
	total_n = t1-t0
	print('CPU matrix mul time {:f}s'.format(total_n))
		