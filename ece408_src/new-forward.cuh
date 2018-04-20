
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

// need for speed, let's hardcode this
#define TOTAL_KERNEL_DATA 5000 // assuming four bytes float
#define KERNEL_LENGTH 1250
#define TILE_WIDTH 28

namespace mxnet
{
namespace op
{

// Define constant memory
__constant__ float const_kernel[50][5][5];

__global__ void forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K) {
    
    // define shared memory for a input image, defined the same as block size
    __shared__ float shared_x[TILE_WIDTH][TILE_WIDTH];

    #define y4d(i3,i2,i1,i0) y[(i3) * (28800) + (i2)*(576) + (i1)*(24) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (784) + (i2)*(784) + (i1)*(28) + i0]

    // allocate workers to load into shared memory
    int b, h, w, m;
    b = blockIdx.x;
    m = blockIdx.y;
    h = threadIdx.y;
    w = threadIdx.x;

    // load input feature map into shared memory
    shared_x[h][w] = x4d(b,0,h,w);
    __syncthreads();

    if(h<24 && w < 24){
        float acc = 0.0f;
        // unrolling + pre-calculating indexes
        // declare index in a row to utilize compiler optimization
        int w_1, w_2, w_3, w_4, h_1, h_2, h_3, h_4;
        w_1 = w+1;
        w_2 = w+2;
        w_3 = w+3;
        w_4 = w+4;
        h_1 = h+1;
        h_2 = h+2;
        h_3 = h+3;
        h_4 = h+4;
  
        // unrolling forloop, helps a lot!
        acc += shared_x[h][w] * const_kernel[m][0][0];
        acc += shared_x[h][w_1] * const_kernel[m][0][1];
        acc += shared_x[h][w_2] * const_kernel[m][0][2];
        acc += shared_x[h][w_3] * const_kernel[m][0][3];
        acc += shared_x[h][w_4] * const_kernel[m][0][4];
        acc += shared_x[h_1][w] * const_kernel[m][1][0];
        acc += shared_x[h_1][w_1] * const_kernel[m][1][1];
        acc += shared_x[h_1][w_2] * const_kernel[m][1][2];
        acc += shared_x[h_1][w_3] * const_kernel[m][1][3];
        acc += shared_x[h_1][w_4] * const_kernel[m][1][4];
        acc += shared_x[h_2][w] * const_kernel[m][2][0];
        acc += shared_x[h_2][w_1] * const_kernel[m][2][1];
        acc += shared_x[h_2][w_2] * const_kernel[m][2][2];
        acc += shared_x[h_2][w_3] * const_kernel[m][2][3];
        acc += shared_x[h_2][w_4] * const_kernel[m][2][4];
        acc += shared_x[h_3][w] * const_kernel[m][3][0];
        acc += shared_x[h_3][w_1] * const_kernel[m][3][1];
        acc += shared_x[h_3][w_2] * const_kernel[m][3][2];
        acc += shared_x[h_3][w_3] * const_kernel[m][3][3];
        acc += shared_x[h_3][w_4] * const_kernel[m][3][4];
        acc += shared_x[h_4][w] * const_kernel[m][4][0];
        acc += shared_x[h_4][w_1] * const_kernel[m][4][1];
        acc += shared_x[h_4][w_2] * const_kernel[m][4][2];
        acc += shared_x[h_4][w_3] * const_kernel[m][4][3];
        acc += shared_x[h_4][w_4] * const_kernel[m][4][4];
  
        y4d(b,m,h,w) = acc;
      }

    #undef y4d
    #undef x4d
}




/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    cudaStream_t s = y.stream_->stream_;
    const int B = x.shape_[0]; // batch size
    const int M = y.shape_[1]; // output channels
    const int C = x.shape_[1]; // input channels
    const int H = x.shape_[2]; // height
    const int W = x.shape_[3]; // width
    const int K = w.shape_[3]; // filter bank size
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    dim3 gridDim(B,M,1);
    cudaMemcpyToSymbol(const_kernel, w.dptr_, KERNEL_LENGTH*sizeof(float));
    forward_kernel<<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,B,M,C,H,W,K);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}


/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif