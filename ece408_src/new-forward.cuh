
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{


template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y, const DType *x, const DType *k, const int B, const int M, const int C, const int H, const int W, const int K) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    //(void)H_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)W_out; // silence declared but never referenced warning. remove this line when you start working
    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    /*
        Your code here!
    */
        //access the whole input image, which is 28*28, by one dimension, x, to achieve memory coalising in load
        //y and z cannot achieve memory coalise
    int b = blockIdx.x*blockDim.x+threadIdx.x;
    int m = blockIdx.y*blockDim.y+threadIdx.y;
    int hw = blockIdx.z*blockDim.z+threadIdx.z;
    int h = hw / W_out;
    int w = hw % W_out;
    if((b < B) && (m < M) && (h < H_out) && (w < W_out)){
        DType pValue = 0;
         for(int c = 0; c < C; ++c){
            for(int p = 0; p < K; ++p){
                for(int q = 0; q < K; ++q){
                    pValue += x4d(b,c,h+p,w+q)*k4d(m,c,p,q);
                    
                }
            }
        }
        y4d(b,m,h,w) = pValue;

    }

    #undef y4d
    #undef x4d
    #undef k4d
}


// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    

    // Use mxnet's CHECK_EQ to do assertions.
    //CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    // cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
 //__global__ void forward_kernel(DType *y, const DType *x, const DType *k, const int B, const int M, const int C, const int H, const int W, const int K) {

    const int _B = x.shape_[0];//???number of memory banks
    const int _M = y.shape_[1];// the number of output feature maps
    //INPUT: X[C, H, W] 3D array
    const int _C = x.shape_[1];//number of input feature maps
    const int _H = x.shape_[2];//height of each input map image
    const int _W = x.shape_[3];//width of each input map image
    const int _K = w.shape_[3];//the height and width of each filter bank K[M,C,K,K]
    
    // //GPU data pointer
    // DType *device_x;//kernel pointer to receive the cuda memory allocation of X
    // DType *device_y;//kernel pointer to receive the cuda memory allocation of Y
    // DType *device_w;//kernel pointer to receive the cuda memory allocation of W


    // //calculate the size of input x, output y and mask w
    // sizeX = _B * _M * _C * _H;
    // sizeW = _C * _K * _K;
    // sizeY = _B * M * (_H - _K + 1) * (_W - _K + 1);

    // //alocate GPU memory
    // //cudaMalloc((void **) &deviceA,sizeA)
    // cudaMalloc((void **) &device_x, sizeX);
    // cudaMalloc((void **) &device_w, sizeW);
    // cudaMalloc((void **) &device_Y, sizeY);

    // //copy x and w to GPU
    // cudaMemcpy(device_x, x , sizeX, cudaMemcpyHostToDevice);
    // cudaMemcpy(device_w, w , sizew, cudaMemcpyHostToDevice);
    
    //initialize grid and block dimension
    
    int blocks_xdimension = (_B)/8 + 1;  
    int blocks_ydimension = (_M)/8 + 1;
    int blocks_zdimension = (( _H - _K + 1) * ( _W - _K + 1))/8 + 1;
    dim3 dimGrid(blocks_xdimension , blocks_ydimension , blocks_zdimension);
    dim3 dimBlock(8,8,8);
    // Call the kernel
    forward_kernel<gpu, DType><<<dimGrid, dimBlock>>>(y.dptr_,x.dptr_,w.dptr_, _B, _M, _C, _H, _W, _K);


    // Set the kernel dimensions
    // dim3 gridDim(0);
    // dim3 blockDim(0);

    // Call the kernel
    // forward_kernel<gpu, DType><<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}



}
}

#endif