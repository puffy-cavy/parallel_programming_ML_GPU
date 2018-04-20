#ifndef MXNET_OPERATOR_NEW_FORWARD_H_
#define MXNET_OPERATOR_NEW_FORWARD_H_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

// This function is called by new-inl.h
// Any code you write should be executed by this function
template <typename cpu, typename DType>
void forward(mshadow::Tensor<cpu, 4, DType> &y, const mshadow::Tensor<cpu, 4, DType> &x, const mshadow::Tensor<cpu, 4, DType> &k)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    The code in 16 is for a single image.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct, not fast (this is the CPU implementation.)
    */

    //OUTPUT: Y[M, H-K+1, W-K+1]

    const int B = x.shape_[0];//???number of memory banks
    const int M = y.shape_[1];// the number of output feature maps
    //INPUT: X[C, H, W] 3D array
    const int C = x.shape_[1];//number of input feature maps
    const int H = x.shape_[2];//height of each input map image
    const int W = x.shape_[3];//width of each input map image
    const int K = k.shape_[3];//the height and width of each filter bank K[M,C,K,K]

// There are MxC filter banks. 
// Filter bank K[M,C,K,K] is used when using input feature map X[C, H, W] to calculate output feature map Y[M, H-K+1, W-K+1]. 
// Note that each output feature map is the sum of convolutions of all input feature maps. 
// Therefore, we can consider the forward propagation path of a convolutional layer as set of M 3D convolutions, 
// where each 3D convolution is specified by a 3D filter bank that is a C x K x K submatrix of W.

    for (int b = 0; b < B; ++b) { //for each image in the batch->for each bank???
        //CHECK_EQ(0, 1) << "Missing an ECE408 CPU implementation!";
        for(int m = 0; m < M; ++m){ //for each output feature maps
          for(int h = 0; h < H; ++h){ //for each output element
            for(int w = 0; w < W; ++w){
              y[b][m][h][w] = 0;
              for(int c = 0; c < C; ++c){ //sum over all input feature maps
                for(int p = 0; p < K; ++p){ //K*K filter
                  for(int q = 0; q < K; ++q){
                    y[b][m][h][w] += x[b][c][h + p][w + q] * k[m][c][p][q];
                  }
                }
              }
            }
          }
        }//b = 10000; m = 50; h = 28 = w; C = 1; K = 5
        // ... a bunch of nested loops later...

    }


}
}
}

#endif
