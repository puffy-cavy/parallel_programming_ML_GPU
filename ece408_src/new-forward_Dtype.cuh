
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 16

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y, const DType *x, const DType *k, const int B, const int M, const int C, const int H, const int W, const int K) {
    
    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    int W_grid = (W_out-1)/TILE_WIDTH + 1;
    int b,m,h,w,c,p,q, h_base, w_base;
    extern __shared__ double shmem[];
    int in_tile_width = TILE_WIDTH + K - 1;
    DType * x_shared = &shmem[0];
    DType * k_shared = &shmem[in_tile_width*in_tile_width];

    b = blockIdx.x;
    m = blockIdx.y;
    h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
    h_base = h - threadIdx.y;
    w_base = w - threadIdx.x;
    double acc = 0.0f;
    for(c = 0; c<C;c++){
      /*
        1. for all channels, loop-through all the input channel
        2. load corresponding kernel
        3. load corresponding input channel
        4. calculation
      */
      if(threadIdx.x < K && threadIdx.y < K){
        k_shared[threadIdx.y*K+threadIdx.x] = k4d(m,c,threadIdx.y,threadIdx.x);
      }
      __syncthreads();

      for(int i = h; i<h_base + in_tile_width && i<H; i++){
        for(int j = w; j<w_base + in_tile_width && j<W; j++){
          x_shared[threadIdx.y*W+threadIdx.x] = x4d(b,c,i,j);
        }
      }
      __syncthreads();
      if(h<H_out && w < W_out){        
        for(p = 0; p < K; p++){
          for(q = 0; q < K; q++){
            acc+=x_shared[(threadIdx.x+p)*in_tile_width+(threadIdx.y+q)]*k_shared[p*K+q];
          }
          __syncthreads();          
        }     
      }
    }
    if(h<H_out && w<W_out)
      y4d(b,m,h,w) = acc;   
    //y4d(w,h,m,b) = acc;
    #undef y4d
    #undef x4d
    #undef k4d
}




// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {


    // Use mxnet's CHECK_EQ to do assertions.
    // CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
      const int B = x.shape_[0]; // batch size
      const int M = y.shape_[1]; // output channels
      const int C = x.shape_[1]; // input channels
      const int H = x.shape_[2]; // height
      const int W = x.shape_[3]; // width
      const int K = w.shape_[3]; // filter bank size

    // Set the kernel dimensions
    const int W_grid = ((W - K + 1) - 1)/TILE_WIDTH + 1;
    const int H_grid = ((H - K + 1) - 1)/TILE_WIDTH + 1;
    const int Z = W_grid * H_grid;
    dim3 blockDim(TILE_WIDTH,TILE_WIDTH,1);
    dim3 gridDim(B,M,Z);


    unsigned long long shared_mem_size = sizeof(double)*((TILE_WIDTH + K - 1)*(TILE_WIDTH + K - 1)+ K * K);
    // Call the kernel
    forward_kernel<gpu, DType><<<gridDim, blockDim, shared_mem_size, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}



}
}

#endif
