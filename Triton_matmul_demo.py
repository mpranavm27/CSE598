import torch
import triton
import triton.language as tl
import argparse
import time
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA Kernel (C++)
# ---------------------------------------------------------------------------

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matmul_cuda_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    auto M = a.size(0);
    auto K = a.size(1);
    auto N = b.size(1);
    
    auto c = torch.zeros({M, N}, torch::device(a.device()).dtype(a.dtype()));
    
    const int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    
    matmul_cuda_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        M, N, K
    );
    
    return c;
}
"""

cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);
"""

# Compile CUDA kernel on the fly
try:
    matmul_cuda_module = load_inline(
        name='matmul_cuda',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['matmul_cuda'],
        with_cuda=True,
        extra_cuda_cflags=["-O2"]
    )
    HAS_CUDA_COMPILER = True
except Exception as e:
    print(f"Warning: Failed to compile CUDA kernel. Error: {e}")
    if "cl" in str(e) or "WinError 2" in str(e):
        print("  -> Hint: On Windows, you need the MSVC compiler (cl.exe) in your PATH.")
        print("     Try running this from the 'x64 Native Tools Command Prompt for VS 20xx'.")
    HAS_CUDA_COMPILER = False


# ---------------------------------------------------------------------------
# Triton Kernel
# ---------------------------------------------------------------------------

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32, GROUP_SIZE_M=8
    )
    return c


# ---------------------------------------------------------------------------
# Benchmark Harness
# ---------------------------------------------------------------------------

def benchmark(size, provider):
    device = torch.device("cuda")
    # Use float32 for CUDA kernel simplicity in this demo, though Triton uses fp16
    dtype = torch.float32 if provider == 'cuda' else torch.float16
    
    x = torch.randn(size, size, device=device, dtype=dtype)
    y = torch.randn(size, size, device=device, dtype=dtype)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(x, y), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(x, y), quantiles=quantiles)
    if provider == 'cuda':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_cuda_module.matmul_cuda(x, y), quantiles=quantiles)
    
    tflops = lambda ms: 2 * size ** 3 / (ms * 1e-3) * 1e-12
    return tflops(ms), tflops(max_ms), tflops(min_ms)


def main():
    parser = argparse.ArgumentParser(description="Triton MatMul Demo")
    parser.add_argument('--min-size', type=int, default=128, help="Minimum matrix size")
    parser.add_argument('--max-size', type=int, default=4096, help="Maximum matrix size")
    parser.add_argument('--step', type=int, default=128, help="Step size")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This demo requires an NVIDIA GPU.")
        return

    print(f"Running benchmark from size {args.min_size} to {args.max_size}...")
    header = f"{'Size':<10} | {'Torch TFLOPS':<15} | {'Triton TFLOPS':<15} | {'Triton Speedup':<15}"
    if HAS_CUDA_COMPILER:
        header += f" | {'CUDA TFLOPS':<15} | {'CUDA Speedup':<15}"
    print(header)
    print("-" * len(header))

    for size in range(args.min_size, args.max_size + 1, args.step):
        try:
            torch_perf, _, _ = benchmark(size, 'torch')
            triton_perf, _, _ = benchmark(size, 'triton')
            
            triton_speedup = triton_perf / torch_perf
            
            row = f"{size:<10} | {torch_perf:<15.4f} | {triton_perf:<15.4f} | {triton_speedup:<15.2f}x"
            
            if HAS_CUDA_COMPILER:
                cuda_perf, _, _ = benchmark(size, 'cuda')
                cuda_speedup = cuda_perf / torch_perf
                row += f" | {cuda_perf:<15.4f} | {cuda_speedup:<15.2f}x"
            
            print(row)
        except Exception as e:
            print(f"{size:<10} | Error: {e}")

if __name__ == "__main__":
    main()
