from numba import cuda, vectorize
import math

@cuda.jit
def sigmoid_activation_forward(Z, out):
    i, j = cuda.grid(2)
    ny, nx = Z.shape
    if i >= ny and j >= nx:
        return
    out[i, j] = 1. / (1 + math.exp(-Z[i, j]))
    
@cuda.jit
def sigmoid_activation_backprop(Z, dA, out):
    i, j = cuda.grid(2)
    ny, nx = Z.shape
    if i >= ny and j >= nx:
        return
    out[i, j] = dA[i, j] * 1. / (1 + math.exp(-Z[i, j])) * (1 - 1. / (1 + math.exp(-Z[i, j])))

@cuda.jit
def relu_activation_forward(A, out):
    i, j = cuda.grid(2)
    ny, nx = A.shape
    if i >= ny and j >= nx:
        return
    if A[i, j] <= 0:
        out[i, j] = 0
    else:
        out[i, j] = A[i, j]

@cuda.jit
def relu_activation_backprop(Z, dA, out):
    i, j = cuda.grid(2)
    ny, nx = Z.shape
    if i >= ny and j >= nx:
        return
    if Z[i, j] > 0:
        out[i, j] = dA[i, j]

@cuda.jit
def linear_activation_forward(A, W, b, out):
    i, j = cuda.grid(2)
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
    pass
    
#Threads per block
TPB = 16
@cuda.jit
def fast_matmul(A, B, C):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    x, y = cuda.grid(2)
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