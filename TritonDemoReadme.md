# Triton Demo Apps

This repository contains standalone demo applications showcasing the performance of OpenAI Triton.

## Prerequisites

- **GPU**: NVIDIA GPU (Compute Capability 7.0+) or AMD GPU (ROCm).
- **OS**: Linux (Triton is officially supported on Linux). Windows support is experimental/limited.
- **Python**: 3.8+
- **CUDA Toolkit**: Required if you want to run the CUDA C++ benchmark (`nvcc` must be in your PATH).

## Installation

1. Install dependencies:
   ```bash
   pip install torch triton
   ```

## Demos

### 1. Vector Addition (`vector_add.py`)
The "Hello World" of Triton. Adds two vectors element-wise.

**Run:**
```bash
python vector_add.py
```

**What it demonstrates:**
- Basic kernel definition (`@triton.jit`)
- Loading/Storing data with masks
- 1D Launch Grid

### 2. Matrix Multiplication (`matmul_demo.py`)
A high-performance Blocked Matrix Multiplication kernel.

**Run:**
```bash
python matmul_demo.py
```

**Options:**
- `--min-size`: Minimum matrix size (default: 128)
- `--max-size`: Maximum matrix size (default: 4096)
- `--step`: Step size for benchmarking (default: 128)

**What it demonstrates:**
- 2D Launch Grid
- Blocked memory access for L2 cache optimization
- Comparison against PyTorch and raw CUDA C++

## Troubleshooting

### Windows: "The system cannot find the file specified" or "cl.exe not found"

If you see an error like `[WinError 2] The system cannot find the file specified` when compiling the CUDA kernel (in `matmul_demo.py`), it means `torch` cannot find the Microsoft Visual C++ compiler (`cl.exe`), which is required to link CUDA code on Windows.

**Solution:**
1.  Ensure you have **Visual Studio Build Tools** installed with the "Desktop development with C++" workload.
2.  Run the demo from the **x64 Native Tools Command Prompt for VS 20xx** (search for this in the Start Menu). This sets up the necessary environment variables for `cl.exe`.
