#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16
#define BLOCK_SIZE 128
//__constant__ float deviceKernel[14112];

namespace mxnet
{
namespace op
{
__global__ void gemm(float *A, float *B, float *C, int numARows, int numAColumns, int numBColumns) {
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float temp = 0;

  for(int m = 0; m < ceil((float)numAColumns/TILE_WIDTH); m++){
    if(Row < numARows && m * TILE_WIDTH + tx < numAColumns)
      subTileA[ty][tx] = A[Row * numAColumns + m * TILE_WIDTH + tx]; //A[Row][m * TILE_WIDTH + tx]
    else
      subTileA[ty][tx] = 0;

    if(m * TILE_WIDTH + ty < numAColumns && Col < numBColumns)
      subTileB[ty][tx] = B[(m * TILE_WIDTH + ty) * numBColumns + Col]; //B[m * TILE_WIDTH + ty][Col]
    else
      subTileB[ty][tx] = 0;
      
    __syncthreads();
    for(int k = 0; k < TILE_WIDTH; k++){
      temp += subTileA[ty][k] * subTileB[k][tx];
    }
    __syncthreads();
  }
  if(Row < numARows && Col < numBColumns)
    C[Row * numBColumns + Col] = temp;
}


__global__ void unroll(const float *x, float *x_unroll, int b, const int C, const int H, const int W, const int K, int H_out, int W_out, int H_unroll, int W_unroll) {
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < C * W_unroll) {
        int c = idx / W_unroll;
        int s = idx % W_unroll;
        int h_out = s / W_out;
        int w_out = s % W_out;
        int w_unroll = h_out * W_out + w_out;
        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                int h_unroll = c * K * K + p * K + q;
                x_unroll[h_unroll * W_unroll + w_unroll] = x4d(b, c, h_out + p, w_out + q);
            }
        }
    }
#undef x4d
}

void convLayer_forward(float *y, float *x, float *k, int B, int M, int C, int H, int W, int K){
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int H_unroll = C * K * K;
    int W_unroll = H_out * W_out;

    dim3 dimGrid1(ceil((float)H_unroll * W_unroll/BLOCK_SIZE), 1, 1);
    dim3 dimBlock1(BLOCK_SIZE, 1, 1);

    //int numARows = M;
    //int numAColumns = H_unroll;
    //int numBColumns = W_unroll;
    //int numBRows = numAColumns;
    //int numCRows = numARows;
    //int numCColumns = numBColumns;
    dim3 dimGrid2(ceil((float)W_unroll/TILE_WIDTH), ceil((float)M/TILE_WIDTH), 1);
    dim3 dimBlock2(TILE_WIDTH, TILE_WIDTH, 1);
    

    float *x_unroll;
    cudaMalloc((void **) &x_unroll, H_unroll * W_unroll * sizeof(float));
    
    for(int b = 0; b < B; b++){
        //unroll(x_unrolled, H_unroll * W_unroll, &x[b*C*H*W], C, K, H, W);
        //gemm(k, x_unrolled, &y[b*M*H_out*W_out], M, H_unroll, W_unroll);
        unroll<<<dimGrid1, dimBlock1>>>(x, x_unroll, b, C, H, W, K, H_out, W_out, H_unroll, W_unroll);
        //unrollKernel<<<dimGrid1, dimBlock1>>>(x_unrolled, H_unroll * W_unroll, &x[b*C*H*W], C, K, H, W);
        gemm<<<dimGrid2, dimBlock2>>>(k, x_unroll, &y[b*M*H_out*W_out], M, H_unroll, W_unroll);
    }
    cudaFree(x_unroll);
}


/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   We only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // // Use mxnet's CHECK_EQ to do assertions.
    // CHECK_EQ(0, 1) 

    const int B = x.shape_[0];
    const int M = y.shape_[1]; // num_filter
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    dim3 gridDim((B + 511) / 512);
    dim3 blockDim(512);
    
    //printf("printing!!!!: %d, %d, %d, %d, %d, %d\n", B, M, C, H, W, K);
    //printf("printing!!!!: %d, %d, %d, %d\n", w.shape_[0], w.shape_[1], w.shape_[2], w.shape_[3]);
    //int kernelLength = M * C * K * K;
    //printf("printing!!!!: %d \n", kernelLength);
    //cudaMemcpyToSymbol(deviceKernel, w.dptr_, kernelLength * sizeof(float)); 

    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    //forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
    convLayer_forward(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    assert(0 && "No forward implementation for other datatypes needed");
}
}
}

#endif
