from numba import cuda, vectorize,  float32
import math

@cuda.jit
def sigmoid_activation_forward(Z, out):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    i = tx + bx * bw
    j = ty + by * bh
    ny, nx = Z.shape
    if i >= ny and j >= nx:
        return
    out[i, j] = 1. / (1 + math.exp(-Z[i, j]))
    
@cuda.jit
def sigmoid_activation_backprop(Z, dA, out):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    i = tx + bx * bw
    j = ty + by * bh
    ny, nx = Z.shape
    if i >= ny and j >= nx:
        return
    if(i < out.shape[0] and j < out.shape[1]):
        out[i, j] = dA[i, j] * 1. / (1 + math.exp(-Z[i, j])) * (1 - 1. / (1 + math.exp(-Z[i, j])))

@cuda.jit
def relu_activation_forward(A, out):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    i = tx + bx * bw
    j = ty + by * bh
    ny, nx = A.shape
    if i >= ny and j >= nx:
        return
    if A[i, j] <= 0:
        out[i, j] = 0
    else:
        out[i, j] = A[i, j]

@cuda.jit
def relu_activation_backprop(Z, dA, out):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    i = tx + bx * bw
    j = ty + by * bh
    ny, nx = Z.shape
    if i >= ny and j >= nx:
        return
    if Z[i, j] > 0:
        out[i, j] = dA[i, j]

@cuda.jit
def linear_activation_forward(A, W, b, out):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    i = tx + bx * bw
    j = ty + by * bh
    ny, nx = W.shape
    if i >= ny and j >= nx:
        return
    tmp = 0.
    if i < out.shape[0] and j < out.shape[1]:
        for k in range(A.shape[1]):
            tmp += A[i, k] * W[k, j]
        out[i, j] = tmp
        out[i, j] += b[j]
    

@cuda.jit
def linear_activation_backprop(dZ, W, out):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    i = tx + bx * bw
    j = ty + by * bh
    nx, ny = W.shape
    if i >= ny and j >= nx:
        return
    tmp = 0.
    # switched indices j <-> k in W
    if i < out.shape[0] and j < out.shape[1]:
        for k in range(dZ.shape[1]):
            tmp += dZ[i, k] * W[j, k]
        out[i, j] = tmp
        
    
#Threads per block
TPB = 16
@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(int(A.shape[1] / TPB)):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp

# CUDA kernel
@cuda.jit
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp