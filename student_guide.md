# Student's Guide to the CUDA-MPI Educational Cluster

This guide will help you get started with the CUDA-MPI educational cluster for your coursework on parallel computing and domain decomposition.

## Overview

The cluster consists of multiple laptops connected via Ethernet, each with an NVIDIA GPU. You'll be able to:

- Write and run CUDA code that executes across multiple GPUs
- Learn and implement domain decomposition techniques
- Execute parallel MPI programs across all nodes
- Share files automatically between all nodes

## Getting Started

### Logging In

Each laptop has one or more user accounts set up. You can log in with:

- Username: (provided by your instructor, usually `student1`, `student2`, etc.)
- Password: (provided by your instructor)

### Where to Write Code

All nodes share the filesystem, so any files you create in:
- Your home directory (`/home/your-username/`)
- The shared directory (`/shared/`)

will be visible on all nodes. This means you can edit your code on one laptop and run it across all laptops.

## Running Your First CUDA-MPI Program

1. Open a terminal

2. Run the built-in example:
   ```bash
   run-example hello 2
   ```
   This compiles and runs a simple CUDA-MPI hello world program on 2 processes.

3. Examine the source code:
   ```bash
   cat /usr/share/doc/cuda-mpi-cluster/examples/cuda-mpi-hello.c
   ```

4. Try the domain decomposition example:
   ```bash
   run-example domain 4
   ```
   This shows how to split work across multiple nodes and GPUs.

## Creating Your Own Programs

### Workflow

1. **Create a new C file**  
   ```bash
   cd ~/mpi-examples
   nano my_program.c
   ```

2. **Compile it with CUDA and MPI support**  
   ```bash
   compile-cuda-mpi my_program.c my_program
   ```

3. **Run across multiple nodes**  
   ```bash
   run-mpi -n 4 ./my_program
   ```
   This runs your program with 4 processes distributed across the available nodes.

### Example MPI Program with CUDA

Here's a simple example that combines MPI and CUDA:

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>

// CUDA kernel that adds a constant to each array element
__global__ void add_constant(float *data, int n, float constant) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += constant;
    }
}

int main(int argc, char **argv) {
    int rank, size, i;
    int n = 1024; // Size of array
    float *data;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Calculate local array size
    int local_n = n / size;
    int remainder = n % size;
    // Distribute remainder elements among first few ranks
    if (rank < remainder) {
        local_n++;
    }
    
    // Allocate and initialize local array
    data = (float*)malloc(local_n * sizeof(float));
    for (i = 0; i < local_n; i++) {
        data[i] = rank * 100.0f + i;
    }
    
    printf("Rank %d: Initial values [%.1f, %.1f, ..., %.1f]\n", 
           rank, data[0], data[1], data[local_n-1]);
    
    // Allocate device memory
    float *d_data;
    cudaMalloc(&d_data, local_n * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_data, data, local_n * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Run kernel (add rank number to each element)
    int block_size = 256;
    int grid_size = (local_n + block_size - 1) / block_size;
    add_constant<<<grid_size, block_size>>>(d_data, local_n, 10.0f);
    
    // Copy back to host
    cudaMemcpy(data, d_data, local_n * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_data);
    
    printf("Rank %d: After GPU processing [%.1f, %.1f, ..., %.1f]\n", 
           rank, data[0], data[1], data[local_n-1]);
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Simple gather to demonstrate results (just first element from each rank)
    float *gathered = NULL;
    if (rank == 0) {
        gathered = (float*)malloc(size * sizeof(float));
    }
    MPI_Gather(&data[0], 1, MPI_FLOAT, gathered, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Print gathered results from rank 0
    if (rank == 0) {
        printf("Gathered first elements: [");
        for (i = 0; i < size; i++) {
            printf("%.1f%s", gathered[i], (i < size-1) ? ", " : "");
        }
        printf("]\n");
        free(gathered);
    }
    
    // Clean up
    free(data);
    MPI_Finalize();
    return 0;
}
```

## Domain Decomposition Patterns

Here are common patterns for domain decomposition:

### 1. Block Decomposition (1D)

Divide a 1D array into contiguous blocks:

```c
// Calculate local array size
int local_n = n / size;
int remainder = n % size;
// Handle uneven division
if (rank < remainder) {
    local_n++;
}
// Calculate starting index
int start_idx = rank * (n / size) + (rank < remainder ? rank : remainder);
```

### 2. Grid Decomposition (2D)

Divide a 2D grid among processes:

```c
// For a width×height grid with size processes
int rows_per_process = height / size;
int my_start_row = rank * rows_per_process;
int my_end_row = (rank == size - 1) ? height : (rank + 1) * rows_per_process;

// Process my portion of the grid
for (int i = my_start_row; i < my_end_row; i++) {
    for (int j = 0; j < width; j++) {
        // Process grid[i][j]
    }
}
```

### 3. Scatter/Gather Pattern

Use MPI collective operations to distribute and collect data:

```c
// Scatter data to all processes
MPI_Scatter(global_data, elements_per_process, MPI_FLOAT,
           local_data, elements_per_process, MPI_FLOAT,
           0, MPI_COMM_WORLD);

// Process local data with CUDA
// ...

// Gather results back
MPI_Gather(local_data, elements_per_process, MPI_FLOAT,
          global_data, elements_per_process, MPI_FLOAT,
          0, MPI_COMM_WORLD);
```

## Troubleshooting

### Common Issues

1. **NFS Connection Issues**
   
   If you can't see shared files, try:
   ```bash
   # Check if NFS mounts are working
   df -h
   # Try to manually remount
   sudo mount -a
   ```

2. **MPI Communication Problems**
   
   If MPI processes can't communicate:
   ```bash
   # Check if all hosts can ping each other
   ping mpi-node1
   # Verify hostfile is correct
   cat /etc/openmpi-hostfile
   ```

3. **CUDA Errors**
   
   For CUDA issues:
   ```bash
   # Check if NVIDIA driver is working
   nvidia-smi
   # Ensure CUDA libraries are in path
   echo $LD_LIBRARY_PATH
   ```

### Getting Help

If you encounter issues:

1. Check documentation: `cat /usr/share/doc/cuda-mpi-cluster/README.md`
2. Look at example code for reference
3. Ask your instructor or TA for assistance

## Advanced Topics

### Performance Optimization

1. **Overlap Communication and Computation**
   
   Use non-blocking MPI calls (`MPI_Isend`, `MPI_Irecv`) to overlap communication with computation.

2. **GPU Stream Concurrency**
   
   Use CUDA streams to overlap data transfers and kernel execution:
   
   ```c
   cudaStream_t stream;
   cudaStreamCreate(&stream);
   kernel<<<grid, block, 0, stream>>>(args);
   cudaMemcpyAsync(dst, src, size, kind, stream);
   ```

3. **Efficient Data Transfers**
   
   Minimize data transfers between CPU and GPU, and between nodes.

### Visualizing Results

For simple visualization, you can output data to CSV files and use tools like Matplotlib (Python) or gnuplot.

Example Python plotting script:

```python
import matplotlib.pyplot as plt
import numpy as np

# Load data from your MPI-CUDA program's output
data = np.loadtxt('output.csv', delimiter=',')

# Create a visualization
plt.figure(figsize=(10, 6))
plt.imshow(data, cmap='viridis')
plt.colorbar(label='Value')
plt.title('Domain Decomposition Results')
plt.savefig('result.png')
plt.show()
```

## Further Resources

1. CUDA Documentation: https://docs.nvidia.com/cuda/
2. MPI Tutorial: https://mpitutorial.com/
3. CUDA-Aware MPI: https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/

Happy parallel computing!
