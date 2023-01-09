#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 32

namespace mxnet
{
namespace op
{

__constant__ float deviceKernel[14112];

__global__ void ConvLayerForward(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define const_k4d(i3, i2, i1, i0) deviceKernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int W_grid = ceil((float)W_out/TILE_WIDTH);
    int H_grid = ceil((float)H_out/TILE_WIDTH);

    int X_TILE_WIDTH = TILE_WIDTH + K - 1;
    extern __shared__ float shmem[];
    float* sharedX = &shmem[0];
    //float* sharedW = &shmem[X_TILE_WIDTH * X_TILE_WIDTH];
    int b = blockIdx.x;
    int m = blockIdx.y;
    int h0 = threadIdx.x;
    int w0 = threadIdx.y;

    int h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    int w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    int h = h_base + h0;
    int w = w_base + w0;

    float acc = 0.0;

    for (int c = 0; c < C; c++) {
        //if(h0 < K && w0 < K)
        //    sharedW[h0 * K + w0] = k4d(m, c, h0, w0);
        //__syncthreads();

        for (int i = h; i < h_base + X_TILE_WIDTH; i += TILE_WIDTH) {
            for (int j = w; j < w_base + X_TILE_WIDTH; j += TILE_WIDTH) {
                if(i < H && j < W)
                    sharedX[(i - h_base) * X_TILE_WIDTH + (j - w_base)] = x4d(b, c, i, j);
                else 
                    sharedX[(i - h_base) * X_TILE_WIDTH + (j - w_base)] = 0; 
            }
        }
        __syncthreads();

        for (int p = 0; p < K; p++) {
            for (int q = 0; q < K; q++) {
                if(h0 + p < X_TILE_WIDTH && w0 + q < X_TILE_WIDTH)
                    //acc += sharedX[(h0 + p) * X_TILE_WIDTH + (w0 + q)] * sharedW[p * K + q];
                    acc += sharedX[(h0 + p) * X_TILE_WIDTH + (w0 + q)] * const_k4d(m, c, p, q);
            }
        }
        __syncthreads();
    }
    if(b < B && m < M && h < H_out && w < W_out)
        y4d(b, m, h, w) = acc;

#undef y4d
#undef x4d
#undef k4d
#undef const_k4d
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

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int W_grid = ceil((float)W_out/TILE_WIDTH);
    int H_grid = ceil((float)H_out/TILE_WIDTH);

    int X_TILE_WIDTH = TILE_WIDTH + K - 1;
    size_t shared_size = sizeof(float) * (X_TILE_WIDTH*X_TILE_WIDTH);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(B, M, H_grid * W_grid);

    cudaMemcpyToSymbol(deviceKernel, w.dptr_, (M*C*K*K) * sizeof(float)); 

    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
    ConvLayerForward<<<dimGrid, dimBlock, shared_size>>>(y.dptr_, x.dptr_, B, M, C, H, W, K);
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
