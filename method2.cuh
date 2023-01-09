#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16
#define BLOCK_SIZE 128

namespace mxnet
{
namespace op
{
__global__ void matrixMultiplyShared_Unroll(float *kernel, float *x, float *y,
                                     int numARows, int numAColumns, int numBColumns,
                                     int C, int K, int H, int W) {
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  int W_out = W - K + 1;
  int w = Col % W_out;
  int h = Col / W_out;
  float temp = 0;

  for(int m = 0; m < ceil((float)numAColumns/TILE_WIDTH); m++){
    if(Row < numARows && m * TILE_WIDTH + tx < numAColumns)
      subTileA[ty][tx] = kernel[Row * numAColumns + m * TILE_WIDTH + tx]; //A[Row][m * TILE_WIDTH + tx]
    else
      subTileA[ty][tx] = 0;

    int h_unroll = m * TILE_WIDTH + ty;
    if(h_unroll < numAColumns && Col < numBColumns){
      int q = h_unroll % K;
      h_unroll /= K;
      int p = h_unroll % K;
      int c = h_unroll / K;
      subTileB[ty][tx] = x[c * (H * W) + (h+p) * (W) + w+q];
    }
    else
      subTileB[ty][tx] = 0;
      
    __syncthreads();
    for(int k = 0; k < TILE_WIDTH; k++){
      temp += subTileA[ty][k] * subTileB[k][tx];
    }
    __syncthreads();
  }
  if(Row < numARows && Col < numBColumns)
    y[Row * numBColumns + Col] = temp;
}



void convLayer_forward(float *y, float *x, float *k, int B, int M, int C, int H, int W, int K){
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int H_unroll = C * K * K;
    int W_unroll = H_out * W_out;
    

    dim3 dimGrid(ceil((float)W_unroll/TILE_WIDTH), ceil((float)M/TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    for(int b = 0; b < B; b++){
        matrixMultiplyShared_Unroll<<<dimGrid, dimBlock>>>(k, &x[b*C*H*W], &y[b*M*H_out*W_out], M, H_unroll, W_unroll,C,K,H,W);
    }

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


    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
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
