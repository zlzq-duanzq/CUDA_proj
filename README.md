## Applied Parallel Programming with GPUs

This project is about the parallel programming and optimaztion the 4D convolutional layers in the LeNet-5. We tried four different methods to accelerate the process using **CUDA**.

- **Unroll + shared-memory Matrix multiply**
- **Kernel fusion for unrolling and matrix- multiplication**
- **Shared Memory convolution**
- **Weight matrix (kernel values) in constant memory**

We also tested different combinations of **TILE_WIDTH** and **BLOCK_SIZE**. And the final Op time is 0.092 + 0.193 = 0.285, which is almost 2 times faster than before.