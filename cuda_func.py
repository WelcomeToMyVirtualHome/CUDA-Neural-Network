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
    for k in range(dZ.shape[1]):
        tmp += dZ[i, k] * W[j, k]
    out[i, j] = tmp
    
    
#Threads per block
TPB = 8
@cuda.jit
def fast_matmul(A, B, C):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    x = tx + bx * bw
    y = ty + by * bh
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x
    if x >= C.shape[0] and y >= C.shape[1]:
        return

    tmp = 0.
    for i in range(bpg):
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]
        cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]
        cuda.syncthreads()
    C[x, y] = tmp